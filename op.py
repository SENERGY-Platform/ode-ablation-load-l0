"""The operator: what it infers per message, and when it retrains.

MLOperator is the machine-learning half of Operator Lib. It loads the model
registered under this pipeline and operator from MLflow, calls infer() for every
message that matches a selector, and calls train() when there is no model yet or
when need_retraining() says so. Training runs on Ray.

This operator forecasts the hourly mean of the grid meter's active power 24 hours
ahead. For every message it answers with the forecast for the hour that begins
24 hours after the message's own hour, and stamps the result with that hour, so
the output carries the time it is about rather than the time it was made.
"""

import datetime
import typing

from mlflow.pyfunc import PyFuncModel, PythonModel

from operator_lib.util import Config, MLOperator, Selector
from operator_lib.util.helpers import TrainMlflowLogger

from training import HORIZON_H, TAIL_HOURS, train_model


class CustomConfig(Config):
    """Deployment configuration, typed.

    The base Config already carries mlflow_url, ray_url and ts_conn. Anything
    added here arrives from the operator's deployment config under the same name,
    with this value as the default.
    """

    # Retrain at most this often, in seconds. A day, so a deployment does not
    # spend its life training.
    retrain_after_s = 86400


def _utc(ts: datetime.datetime) -> datetime.datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=datetime.timezone.utc)
    return ts.astimezone(datetime.timezone.utc)


def _as_float(value) -> typing.Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _python_model(model):
    """The PythonModel behind a loaded pyfunc, so its forecast() can be called directly."""
    unwrap = getattr(model, "unwrap_python_model", None)
    if unwrap is not None:
        try:
            return unwrap()
        except Exception:
            pass
    impl = getattr(model, "_model_impl", None)
    inner = getattr(impl, "python_model", None)
    return inner if inner is not None else model


class Operator(MLOperator):
    configType = CustomConfig

    # Which inputs this operator accepts. "args" are the mapping destinations the
    # pipeline is configured with, and the name is what infer() receives as
    # "selector", so one operator can treat several input shapes differently.
    selectors = [
        Selector({"name": "value", "args": ["value"]}),
    ]

    def init(self, *args, **kwargs):
        # State first: under a data split, super().init() trains and replays the
        # test window before it returns, so anything set after it is missing
        # during the replay and overwrites what train() just recorded.
        self.trained_at: typing.Optional[datetime.datetime] = None
        # Completed hourly means of the input, keyed by aware UTC hour start.
        self._hourly: typing.Dict[datetime.datetime, float] = {}
        # The hour being accumulated now.
        self._acc_hour: typing.Optional[datetime.datetime] = None
        self._acc_sum = 0.0
        self._acc_count = 0
        # The model whose tail has been merged, and the forecast cached per hour.
        self._seeded_from = None
        self._cached_hour: typing.Optional[datetime.datetime] = None
        self._cached_target: typing.Optional[datetime.datetime] = None
        self._cached_prediction: typing.Optional[float] = None
        super().init(*args, **kwargs)

    def _seed(self, pm) -> None:
        """Merge the hours the model carries, without overriding what the stream saw."""
        if pm is self._seeded_from:
            return
        self._seeded_from = pm
        self._cached_hour = None
        for key, value in getattr(pm, "tail", {}).items():
            try:
                hour = _utc(datetime.datetime.fromisoformat(key))
            except ValueError:
                continue
            self._hourly.setdefault(hour, float(value))

    def _accumulate(self, hour: datetime.datetime, value: typing.Optional[float]) -> None:
        if self._acc_hour is None:
            self._acc_hour = hour
        elif hour > self._acc_hour:
            if self._acc_count:
                self._hourly[self._acc_hour] = self._acc_sum / self._acc_count
            self._acc_hour = hour
            self._acc_sum, self._acc_count = 0.0, 0
            # Keep memory bounded: only the lags the forecast reads.
            horizon = hour - datetime.timedelta(hours=TAIL_HOURS)
            for old in [h for h in self._hourly if h < horizon]:
                del self._hourly[old]
        elif hour < self._acc_hour:
            # A late message for an hour already closed: leave the closed hour as it is.
            return
        if value is not None:
            self._acc_sum += value
            self._acc_count += 1

    def infer(
        self,
        model: typing.Optional[PyFuncModel],
        data: typing.Dict[str, typing.Any],
        selector: str,
        device_id: str,
        timestamp: datetime.datetime,
    ) -> typing.Tuple[
        typing.Optional[datetime.datetime], typing.Optional[typing.Any], typing.Optional[PythonModel]
    ]:
        """Called for every message. Returns (result timestamp, result, new model).

        The result timestamp is the start of the forecast hour, 24 hours after the
        hour this message falls in.
        """
        if model is None:
            return None, None, None

        pm = _python_model(model)
        self._seed(pm)

        ts = _utc(timestamp)
        hour = ts.replace(minute=0, second=0, microsecond=0)
        self._accumulate(hour, _as_float(data.get("value")))

        # Only complete hours feed the forecast, so it is the same for every
        # message of one hour and is computed once.
        if self._cached_hour != hour:
            target = hour + datetime.timedelta(hours=HORIZON_H)
            if hasattr(pm, "forecast"):
                prediction = pm.forecast(target, self._hourly)
            else:
                prediction = model.predict({"target_hour": target, "hourly": self._hourly})
            self._cached_hour = hour
            self._cached_target = target
            self._cached_prediction = float(prediction)

        return self._cached_target, {"prediction": self._cached_prediction}, None

    def train(
        self, model: typing.Optional[PyFuncModel], logger: TrainMlflowLogger
    ) -> typing.Optional[PythonModel]:
        """Called when there is no model, or when need_retraining() said so.

        Runs inside a Ray session the library opened, with an MLflow run already
        started — so params and metrics logged through "logger" land on the run
        the resulting model is registered from.
        """
        self.trained_at = datetime.datetime.now(datetime.timezone.utc)
        return train_model(logger)

    def need_retraining(self, model: typing.Optional[PyFuncModel]) -> bool:
        """Called after every inference, so it has to be cheap.

        Time-based: a daily retrain refreshes the weekly profile and the
        coefficients. The lags themselves come from the stream between retrains.
        """
        if model is None:
            return True
        if self.trained_at is None:
            return False
        age = datetime.datetime.now(datetime.timezone.utc) - self.trained_at
        return age.total_seconds() >= self.config.retrain_after_s
