"""Training, on Ray.

Separate from op.py because the two run in different places: op.py runs in the
operator's own process for every message, while this runs distributed and rarely.

The task: forecast the hourly mean of the grid meter's active power 24 hours
ahead. The hourly mean is taken over messages, which is how the evaluation forms
its actual value per bucket, so training and scoring agree on what an hour is.

The model is a linear regression on five features of the target hour T, all of
them known 24 hours before T begins:

  lag_last  the last complete hour before the forecast is issued (T - 25h)
  lag_48    the same clock hour two days before T
  lag_168   the same clock hour one week before T
  mean_24   the mean of the 24 complete hours before issue (T - 48h .. T - 25h)
  profile   the mean of that local hour of the week over the training window

A missing lag falls back to the profile, in training and at inference alike.
"""

import datetime
import math
import typing

import ray
from mlflow.pyfunc import PythonModel

from operator_lib.util.helpers import provide_historic_data, TrainMlflowLogger


# How much history one training pass reads.
TRAINING_WINDOW = datetime.timedelta(days=120)
# How far ahead the forecast looks, in hours.
HORIZON_H = 24
# Household routines follow local time, so the weekly profile is keyed on it.
LOCAL_TZ = "Europe/Berlin"
# Hourly means the model carries, so inference has lags from its first message.
TAIL_HOURS = 200
# The last part of the window held out to compare against simple baselines.
HOLDOUT_DAYS = 14

FEATURES = ("lag_last", "lag_48", "lag_168", "mean_24", "profile")


def hour_of_week(hour_utc: datetime.datetime) -> int:
    from zoneinfo import ZoneInfo

    if hour_utc.tzinfo is None:
        hour_utc = hour_utc.replace(tzinfo=datetime.timezone.utc)
    local = hour_utc.astimezone(ZoneInfo(LOCAL_TZ))
    return local.weekday() * 24 + local.hour


class OdeAblationLoadL0Model(PythonModel):
    """The model MLflow registers and op.py later loads.

    Plain lists, dicts and floats only, so MLflow can store it.
    """

    def __init__(self, coef, profile, fallback, tail) -> None:
        self.coef = [float(c) for c in coef]  # intercept, then one per FEATURE
        self.profile = [float(p) for p in profile]  # 168 values, local hour of week
        self.fallback = float(fallback)
        # ISO-8601 UTC hour start -> hourly mean, the last hours before training end.
        self.tail = {str(k): float(v) for k, v in tail.items()}

    def features(self, target_hour: datetime.datetime, hourly: typing.Mapping) -> typing.List[float]:
        """target_hour is an aware UTC hour start; hourly maps aware UTC hour starts to means."""
        prof = self.profile[hour_of_week(target_hour)]

        def lag(hours: int) -> float:
            value = hourly.get(target_hour - datetime.timedelta(hours=hours))
            return prof if value is None else value

        window = [
            hourly.get(target_hour - datetime.timedelta(hours=h))
            for h in range(HORIZON_H + 1, HORIZON_H + 25)
        ]
        window = [v for v in window if v is not None]
        mean_24 = sum(window) / len(window) if len(window) >= 12 else prof
        return [lag(HORIZON_H + 1), lag(48), lag(168), mean_24, prof]

    def forecast(self, target_hour: datetime.datetime, hourly: typing.Mapping) -> float:
        x = self.features(target_hour, hourly)
        y = self.coef[0] + sum(c * v for c, v in zip(self.coef[1:], x))
        if not math.isfinite(y):
            return self.fallback
        return float(y)

    def predict(self, context, model_input=None, params=None):
        # The pyfunc signature carries a context when MLflow calls it and not when
        # the model is called directly, so the payload is taken from whichever holds it.
        payload = model_input if model_input is not None else context
        return self.forecast(payload["target_hour"], payload["hourly"])


def _design(y, profile_by_how, how):
    """Feature matrix for every hour of the regular hourly series y."""
    import numpy as np
    import pandas as pd

    prof = pd.Series(np.asarray(profile_by_how)[how], index=y.index)
    lag_last = y.shift(HORIZON_H + 1).fillna(prof)
    lag_48 = y.shift(48).fillna(prof)
    lag_168 = y.shift(168).fillna(prof)
    mean_24 = y.shift(HORIZON_H + 1).rolling(24, min_periods=12).mean().fillna(prof)
    X = np.column_stack([
        np.ones(len(y)), lag_last.values, lag_48.values, lag_168.values, mean_24.values, prof.values,
    ])
    return X


def _profile(y, how, fallback):
    import numpy as np

    sums = np.zeros(168)
    counts = np.zeros(168)
    values = y.values
    ok = ~np.isnan(values)
    np.add.at(sums, how[ok], values[ok])
    np.add.at(counts, how[ok], 1)
    profile = np.full(168, fallback)
    has = counts > 0
    profile[has] = sums[has] / counts[has]
    return profile


def _rmse(a, b):
    import numpy as np

    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


@ray.remote
def _fit(datasets: typing.List[typing.Any]) -> typing.Optional[dict]:
    import numpy as np
    import pandas as pd

    frames = []
    for dataset in datasets:
        if isinstance(dataset, ray.ObjectRef):
            dataset = ray.get(dataset)
        frame = dataset.to_pandas()
        if "value" in frame.columns and len(frame):
            frames.append(frame[["time", "value"]])
    if not frames:
        return None

    raw = pd.concat(frames, ignore_index=True)
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw["time"] = pd.to_datetime(raw["time"], utc=True)
    raw = raw.dropna()
    if raw.empty:
        return None

    # Mean over messages per UTC hour; hours without a message stay NaN.
    y = raw.set_index("time")["value"].sort_index().resample("1h").mean()
    local = y.index.tz_convert(LOCAL_TZ)
    how = np.asarray(local.dayofweek * 24 + local.hour)
    fallback = float(np.nanmean(y.values))

    # Rows usable for fitting: a target exists and a full week of lags lies before it.
    usable = y.notna().values & (np.arange(len(y)) >= 168)

    stats = {"hours": int(len(y)), "hours_observed": int(y.notna().sum()), "rows_usable": int(usable.sum())}

    # Holdout: fit on everything before the last HOLDOUT_DAYS, score on them.
    split = len(y) - HOLDOUT_DAYS * 24
    if split > 168 * 2:
        y_train = y.copy()
        y_train.iloc[split:] = np.nan
        prof_h = _profile(y_train, how, float(np.nanmean(y_train.values)))
        X_h = _design(y, prof_h, how)  # lags may read holdout hours, as inference does
        fit_rows = usable & (np.arange(len(y)) < split)
        test_rows = usable & (np.arange(len(y)) >= split)
        if fit_rows.sum() > 50 and test_rows.sum() > 24:
            coef_h, *_ = np.linalg.lstsq(X_h[fit_rows], y.values[fit_rows], rcond=None)
            actual = y.values[test_rows]
            stats["holdout_rmse_model"] = _rmse(X_h[test_rows] @ coef_h, actual)
            stats["holdout_rmse_profile"] = _rmse(X_h[test_rows][:, 5], actual)
            stats["holdout_rmse_same_hour_last_week"] = _rmse(X_h[test_rows][:, 3], actual)
            stats["holdout_hours"] = int(test_rows.sum())

    profile = _profile(y, how, fallback)
    X = _design(y, profile, how)
    if usable.sum() > 50:
        coef, *_ = np.linalg.lstsq(X[usable], y.values[usable], rcond=None)
        stats["train_rmse_model"] = _rmse(X[usable] @ coef, y.values[usable])
    else:
        # Too little history to regress: forecast the profile alone.
        coef = np.zeros(len(FEATURES) + 1)
        coef[-1] = 1.0

    observed = y.dropna().iloc[-TAIL_HOURS:]
    tail = {ts.isoformat(): float(v) for ts, v in observed.items()}

    return {
        "coef": [float(c) for c in coef],
        "profile": [float(p) for p in profile],
        "fallback": fallback,
        "tail": tail,
        "stats": stats,
    }


def train_model(logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
    """Read the history, fit, and hand back a model for MLflow to register."""
    with logger.trace("read history"):
        datasets = provide_historic_data(TRAINING_WINDOW)
    if not datasets:
        # Explicitly nothing rather than a model fitted on no data: returning None
        # leaves the previously registered model in place.
        return None

    with logger.trace("fit"):
        fitted = ray.get(_fit.remote(datasets))
    if fitted is None:
        return None

    logger.log_params({
        "training_window_days": TRAINING_WINDOW.days,
        "horizon_h": HORIZON_H,
        "features": ",".join(FEATURES),
        "local_tz": LOCAL_TZ,
        "holdout_days": HOLDOUT_DAYS,
    })
    metrics = {k: float(v) for k, v in fitted["stats"].items()}
    for name, c in zip(("intercept",) + FEATURES, fitted["coef"]):
        metrics[f"coef_{name}"] = float(c)
    logger.log_metrics(metrics)

    return OdeAblationLoadL0Model(
        coef=fitted["coef"],
        profile=fitted["profile"],
        fallback=fitted["fallback"],
        tail=fitted["tail"],
    )
