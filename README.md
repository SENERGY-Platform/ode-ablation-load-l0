# ode-ablation-load-l0

An analytics operator for the SENERGY platform, scaffolded by the Operator
Development Environment. Every file here is yours to change, including this one.

## Summary: 24-hour-ahead forecast of net grid consumption

**Task.** Forecast the net electrical consumption from the grid 24 hours ahead,
as hourly mean power in watts. Success was judged by `evaluation.yaml`: RMSE at
1 h resolution over the test window, threshold 6.1 W.

**Status: not complete.** The one evaluation run scored an RMSE of 223.7 W over
September 2026. The threshold of 6.1 W is not met, and nothing found so far
suggests a credible path to it for this series (see "What the result says").

### How the data was found

The session ran at exposure tier L0 throughout: metadata only, no values.

- The ontology search for Get-Power on the Grid aspect found no device type and
  no import instance. The only import type carrying grid purchase (SolarEdge)
  needs an API key and has no deployed instance.
- A device name search found **Stromzähler**, an Iskra MT 175 grid meter read
  through Tasmota, device
  `urn:infai:ses:device:8ae74f8f-07c9-42c2-ad1c-701b503adac1`, service
  `Get Energy Consumption`
  (`urn:infai:ses:service:61447a7c-5e75-48dd-b2db-bcfcd23fa33d`), variable
  `sensor.MT175.P`. It is active power in W, annotated with the Electricity
  aspect only, which is why the Grid search missed it. That gap in the ontology
  is worth fixing in the device type.
- Its stored history runs from 2024-11-29 to the session's training end
  (2026-09-01). It publishes every few seconds.
- This series was confirmed as the data selection and target. No simulation and
  no new import were needed, because real data existed.

### How Operator Lib scores this

Read from Operator Lib's source (`util/op_ml.py`) before writing any code:

- Each replayed message's prediction is placed in the 1 h bucket of its result
  timestamp.
- The actual value is the mean of the input messages in that bucket.
- Squared errors are averaged within each bucket first, then the RMSE is taken
  over the buckets.

A 24-hour-ahead forecast therefore has to stamp its result with the hour it
forecasts, not the hour it was made.

### The model (`training.py`, `op.py`)

- **Hourly target.** The mean over messages per UTC hour, which is how the
  scorer forms its actual values.
- **Features** of the target hour T, all of them known 24 hours before T begins:
  - the last complete hour (T − 25 h)
  - the same hour 48 h before
  - the same hour 168 h before
  - the mean of the 24 complete hours before issue
  - the mean for that local (Europe/Berlin) hour of the week
- **Fit.** Ordinary least squares, which matches an RMSE criterion. A missing
  lag falls back to the profile, in training and at inference alike.
- **Training window.** 120 days.
- **Logged with every training run.** A 14-day holdout comparing the model with
  two baselines, the weekly profile alone and the same hour last week.
- **Inference.** `infer()` keeps hourly means from the stream, seeded with the
  last 200 training hours the model carries. Each message gets a forecast for
  the hour that starts 24 h after the message's own hour, and the result is
  stamped with that hour. The forecast uses complete hours only, so it is
  computed once per hour.
- **Checked before launch.** The code was run end to end on synthetic data in
  the developer's pod, with no platform reads, so a bug would not cost a launch.

### What the result says

Run `451c810d53dc40899f3a27ded45de73c`, commit `362d328`. The data split was
confirmed, and 648 hourly buckets were scored.

| Forecast | RMSE |
|---|---|
| September test window (the evaluation) | 223.7 W |
| Fitted model, training-side holdout | 246.3 W |
| Weekly profile alone, holdout | 243.8 W |
| Same hour last week, holdout | 338.0 W |

- The test RMSE matches the holdout, so nothing leaked from the test window and
  the code does what was intended.
- The fit put almost all its weight on the weekly profile (coefficient 1.07).
  The lag terms added nothing.
- This meter's day-ahead predictable part is its weekly shape. Roughly 220 W of
  hour-to-hour variation around that shape is not explained by its own past.

A threshold of 6.1 W would require explaining nearly all of that variation a
day in advance. One way to score close to it would be to stamp each forecast
with the current hour, which forecasts nothing ahead. That was deliberately not
done.

### What was left open

- **A profile of `sensor.MT175.P`.** Proposed for the 2026-05-03 to 2026-09-01
  training window, to check its unit and scale, spread, spikes, periodicity and
  gaps. It needs tier L1, and the session stayed at L0, so it was not run. This
  is the step that decides between changing the model and revisiting the
  threshold.
- **A change made without seeing the data.** Drop the lag terms and train on
  365 days. Expected gain: a few watts at most.
- **Weather inputs.** The platform's weather imports (Open-Meteo, yr.no, DWD)
  could be added as inputs. They are unlikely to close a gap of this size.
- **The threshold.** Whether 6.1 W suits this meter, or was set for a
  different signal or scale, is the developer's decision.

## Layout

| File | What it is |
|---|---|
| "main.py" | Entry point of the deployed operator. Hands the process to Operator Lib. |
| "train.py" | Entry point of an experiment. Trains through Operator Lib, then exits. |
| "op.py" | The operator: "infer", "train", "need_retraining", and its config. |
| "training.py" | The Ray training pass and the model MLflow registers. |
| "pyproject.toml" | Dependencies, with Operator Lib pinned at "v1.8.2". |
| "uv.lock" | The resolved dependencies. Written by the scaffold; refresh it yourself. See below. |
| "Dockerfile" | The image. Built by CI; buildable by hand. |
| ".github/workflows/build.yml" | Builds and pushes "ghcr.io/senergy-platform/ode-ablation-load-l0". Change the registry here. |
| "operator.yaml" | What the analytics stack registers: inputs, outputs, config. |
| "evaluation.yaml" | Your criteria for whether a run is good, plus what Operator Lib needs to score a test window itself. ODE never writes this. |

## The lock file

The scaffold ran "uv lock" for you and "uv.lock" is in this working copy, uncommitted
like everything else here. Commit it with the rest.

Refresh it whenever you change a dependency in "pyproject.toml", and commit the two
together:

    uv lock

An experiment runs "uv run python train.py" on the cluster, and uv builds the
environment from "pyproject.toml" and this file — on the Ray head for the driver and
on each worker node for the tasks, out of its own cache.

Without a lock file uv resolves at run time, which works and is worse in one
specific way: the run records a commit SHA as the code that produced it, and two
runs of the same commit can then resolve different dependency versions. The lock
file is what makes the recorded SHA describe the whole run rather than only its
source. That is why it is not left to be remembered — and if the scaffold reported
that it could not write one, the command above is the repair.

## Building by hand

    docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t ghcr.io/senergy-platform/ode-ablation-load-l0:dev .

## The Operator Lib pin

"pyproject.toml" pins Operator Lib at "v1.8.2", the newest at the time
this repository was scaffolded. The library tracks latest and promises no
stability, so moving the pin is a deliberate edit — change it, run "uv lock", and
commit the two together.
