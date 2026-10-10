# Roadmap

What is shipped, what is being considered, and what is deliberately out of scope. This is a
statement of intent, not a commitment: items move as evidence and time allow. Dated history lives in
[CHANGELOG.md](CHANGELOG.md).

## Shipped

- **Decision chain** — detection → ByteTrack tracking → in-path filtering → ground-plane ranging →
  dual-source TTC (calibration-free scale TTC fused with distance TTC) → speed-adaptive risk state
  machine (`NORMAL → ATTENTION → FCW → AEB_WARNING`).
- **Detector-agnostic contract** — `BaseDetector`; the YOLOv8 (`yolo` extra) and D-FINE (`dfine`
  extra) backends both satisfy it, and the decision chain consumes only
  `[x1, y1, x2, y2, score, class]`.
- **Compiler-free install path** — a pure-numpy `bbox_overlaps` shim plus the `numpy>=1.24` alias
  restoration, so the vendored ByteTrack runs on a stock modern Python without MSVC or any C++
  toolchain. (`pip install lapx` covers the `lap` wheel gap on platforms without one.)
- **Per-video self-calibration** — joint `f·H` and `horizon_y` estimation from GPS ego speed and
  calibration-free scale TTC.
- **Zero-data regression suite** — no video, no weights, no network; plus CI on Ubuntu and Windows ×
  Python 3.10 / 3.12, a dedicated lint job, and a full-install job that resolves the optional backend.
- **Packaging** — `pip install mono-aeb`, with sdist and wheel attached to each GitHub Release.
- **Benchmarks** — `benchmarks/synth_tracking.py` (synthetic multi-target tracking regression) and
  `benchmarks/bbox_overlaps_bench.py` (the numpy shim's cost versus the compiled path).
- **Container recipe** — a `Dockerfile` that installs the package with the `dev` extra and runs the
  zero-data suite (`docker build -t mono-aeb .`). No compiler is needed, and no image is published to
  a registry yet.

## Being considered

Roughly ordered by expected value, not by date.

- **Console entry points** — `mono-aeb --check` / `--demo` / `--calibrate`, so the zero-data demo can
  be run without cloning the repository. Today the same commands are `python examples/quickstart.py`.
- **Publishing the container image** — the `Dockerfile` exists and builds locally; pushing it to a
  registry would make `docker run` a one-liner for users.
- **An ONNX Runtime backend** — would make the `--check` and demo paths usable without `torch` /
  `ultralytics`, at some accuracy cost against the current backends.
- **Confidence handling for scale TTC** — the estimate degrades when the closing speed is small and
  the target's box barely changes; a documented usable regime and an explicit confidence gate would
  be more useful than a silent number.
- **Failure-case gallery** — a short, honest set of the situations the chain gets wrong, with the
  inputs behind them.

## Out of scope

- **Vehicle control or actuation.** This is an offline and simulation-verified decision chain. It has
  no functional-safety certification and is not calibrated for a real vehicle; do not drive with it.
- **Shipping model weights or BDD100K data.** The finetuned weights are derived from BDD100K and are
  not redistributable; see "Weights & data" in [README.md](README.md).
- **A hard dependency on `torch` or `ultralytics`.** The default install stays light; heavy backends
  remain optional extras.
- **Replacing the vendored tracker subset with a full upstream copy.** The subset is intentionally
  minimal and kept diffable against upstream.

## Known limitations

- Ground-plane ranging assumes a flat road; strong pitch, roll or fisheye distortion break the model's
  premise.
- Scale TTC needs a few frames of measurable size change — it is unreliable for near-stationary
  relative motion or when a box is clipped by the frame edge.
- `lap` has no wheel for some platform / Python combinations; install `lapx` instead (it provides the
  module name `lap`, so no code change is needed).
- No accuracy figures are claimed in this repository. It ships code and a zero-data regression suite;
  any evaluation number depends on the dataset, the detector and the per-video calibration used.
