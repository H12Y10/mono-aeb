# mono-aeb

[![CI](https://github.com/H12Y10/mono-aeb/actions/workflows/ci.yml/badge.svg)](https://github.com/H12Y10/mono-aeb/actions/workflows/ci.yml)

A detector-agnostic monocular **AEB (Autonomous Emergency Braking) / FCW (Forward Collision Warning)** decision chain: detection → tracking → in-path filtering → ranging → TTC → risk decision.

The two core strengths are **calibration-free scale TTC** — which relies only on the rate of change of a target's image size and does not depend on camera intrinsics or distortion calibration — and **speed-adaptive risk thresholds**, where the AEB trigger point shifts with ego speed. The detector is pluggable: the YOLOv8 backend (`yolo` extra) and the high-accuracy D-FINE backend (`dfine` extra) both implement the same `BaseDetector` interface, and the decision chain consumes only `[x1, y1, x2, y2, score, class]`.

> 中文说明见 [README.zh-CN.md](README.zh-CN.md)。

---

## Features

- **Detector-agnostic contract** — the decision chain is decoupled from any specific detector and works with the 7 AEB classes (person / rider / car / truck / bus / bike / motor).
- **Calibration-free scale TTC** — `τ = s / ṡ`, estimated via a Theil–Sen robust fit of `(t, 1/s)`; no camera calibration required.
- **Distance TTC + fusion** — a ground-plane ranging model recovers closing speed, fused with scale TTC by taking the minimum, with a `persist_k` frame hold to suppress jitter.
- **Speed-adaptive thresholds** — `τ_AEB(v) = t_react + v / (2·a_max)`, with fixed lead margins for FCW / ATTENTION on top; the whole decision gradient shifts with ego speed.
- **Per-video self-calibration** — jointly estimates the ranging scale `f·H` and the horizon `horizon_y` from GPS ego speed + calibration-free scale TTC, removing the per-video camera-parameter inconsistency in BDD100K.
- **Bundled ByteTrack** — the tracker is vendored into the package (`aeb/tracker/vendor/bytetrack/`) and distributed with the wheel; no separate installation needed.

## Pipeline

```
video frame → detection (YOLOv8 / D-FINE)
            → tracking (ByteTrack)
            → in-path filtering (fixed trapezoid ROI, ego-path corridor projection)
            → ranging (ground-plane model, D = f·H / (y_bottom − horizon))
            → TTC (scale τ and distance τ fused via min)
            → decision (state machine NORMAL → ATTENTION → FCW → AEB_WARNING)
```

The risk state machine only escalates while a target is inside the ego-path corridor (`in_path = True`); targets outside the path stay at `NORMAL`, avoiding false triggers from adjacent-lane vehicles.

---

## Installation

Requires Python ≥ 3.10.

```bash
pip install -e .
```

Runtime dependencies:

| Dependency | Purpose | Notes |
| --- | --- | --- |
| `numpy` / `scipy` | numeric computation / Theil–Sen fit | `scipy` is also a transitive dependency of ByteTrack |
| `opencv-python` | image I/O, ROI, visualization | |
| `lap` | ByteTrack matching (linear assignment) | see below |

**About `lap`**: `lap` is the linear-assignment solver used by ByteTrack matching. If no `lap` wheel exists for your platform/Python, install `lapx` instead — it is an alternative distribution of `lap` that ships a module named `lap`:

```bash
pip install lapx
```

Note: this project does **not** fall back in code. When `lap` is missing, installing `lapx` as above is enough (it provides the module name `lap`, so no code changes are needed).

ByteTrack itself is vendored as a **minimal subset** (`aeb/tracker/vendor/bytetrack/`, MIT License) and installed with the wheel, so both `pip install -e .` and a regular `pip install` work without fetching the upstream repository. To use an external/full ByteTrack instead:

| Environment variable | Meaning |
| --- | --- |
| `BYTETRACK_ROOT` | ByteTrack repository root (containing `yolox/tracker/`), overriding the bundled vendor default |

If `pip install -e .` aborts because no `lap` wheel is available for your platform, install `lapx` first, then install this package while skipping dependency resolution:

```bash
pip install lapx
pip install -e . --no-deps
```

**Optional high-accuracy backend D-FINE** additionally requires the PyTorch ecosystem:

```bash
pip install -e ".[dfine]"   # torch / torchvision / pillow
```

**About Windows**: `torch` requires the Microsoft Visual C++ runtime; on some machines (especially lab/clean-room environments) its absence shows up as `OSError: [WinError 126] ... c10.dll` failing to load. Installing the [VC++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) once resolves it. (The `--demo` zero-data demo does not import torch, so it does not need this.)

On non-Chinese Windows locales (console codepage not UTF-8), the example scripts print Chinese and throw `UnicodeEncodeError`; run `set PYTHONUTF8=1` first. Linux/macOS and Chinese Windows are unaffected.

---

## Quick start

### Zero-data demo (no video / no weights)

```bash
python examples/quickstart.py --demo
```

This command drives a synthetic detector that generates a target closing at constant speed, and runs the full detection → tracking → ranging → TTC → decision chain, printing state changes frame by frame:

```
[quickstart] ego speed = 12 m/s, AEB threshold = 1.37s
[quickstart] target closing at a constant 10 m/s; printing the risk state per frame:
  frame   0: NORMAL     d= 45.0m  ttc=  infs
  frame  39: ATTENTION  d= 32.0m  ttc= 3.36s
  frame  69: FCW        d= 22.0m  ttc= 2.36s
  frame  99: AEB        d= 12.1m  ttc= 1.36s
[quickstart] done, 120 frames
```

Available options: `--out video.mp4` (write a visualization video), `--show` (live window), `--frames` / `--fps` / `--v-close`. `--demo` mode does **not** require `ultralytics` / `torch`; it depends only on numpy / scipy / opencv-python / lap.

Dependency self-check:

```bash
python examples/quickstart.py --check
```

### Real video

```bash
python examples/demo_aeb.py --source video.mp4 --coco --out out.mp4
```

- `--coco`: maps COCO classes to the 7 AEB classes when using COCO-pretrained weights (e.g. `yolov8n.pt`); omit it when using AEB-finetuned weights.
- `--ego-speed`: ego speed (m/s); when omitted, GPS speed is read from BDD100K `samples-1k/info/*.json` automatically, falling back to `12 m/s` when unavailable.
- `--detector dfine`: switch to the D-FINE backend (see below).

You may also pass an image directory via `--source ./frames_dir`.

---

## Detector backends

### YOLOv8 (optional backend, `yolo` extra)

```bash
pip install -e ".[yolo]"    # ultralytics (AGPL-3.0)
```

```python
from aeb.detectors import YOLOv8Detector, COCO_TO_AEB

det = YOLOv8Detector("yolov8n.pt", conf=0.25, device="0",
                     class_map=COCO_TO_AEB)  # class_map=None for finetuned weights
```

### Optional high-accuracy backend: D-FINE

D-FINE depends on its upstream repository, located via environment variables to avoid hardcoding local paths:

| Environment variable | Meaning |
| --- | --- |
| `DFINE_ROOT` | D-FINE upstream repository root (containing `src/`, `configs/`) |
| `MONO_AEB_DFINE_WEIGHTS` | path to finetuned weights (`.pth`) |

```bash
export DFINE_ROOT=/path/to/D-FINE
export MONO_AEB_DFINE_WEIGHTS=/path/to/best_stg1.pth
python examples/demo_aeb.py --source video.mp4 --detector dfine
```

Or pass the arguments directly in code:

```python
from aeb.detectors import DFineDetector
det = DFineDetector(weights="/path/to/best_stg1.pth")  # config defaults to dfine_hgnetv2_m_aeb.yml
```

> ⚠️ **Note**: D-FINE's finetuned weights are trained on BDD100K and are subject to that dataset's license (see below); they are not distributed with this repository.

---

## Weights & data

> ⚠️ **Compliance note**: this repository does not contain any model weights or BDD100K videos/annotations.

- **COCO-pretrained YOLOv8 weights** (e.g. `yolov8n.pt`) are provided by ultralytics and obtained under ultralytics' license.
- **AEB-finetuned weights** (YOLOv8 or D-FINE) are finetuned on BDD100K, whose license is "academic/non-commercial use only, no redistribution of raw data". These weights are therefore not published here; they are distributed on a restricted basis (on request). Anyone with legitimate data and license access can reproduce them with this repository's training configuration.

---

## Project structure

```
mono-aeb/
├── aeb/
│   ├── config.py           # camera / ego-path / risk thresholds / detection & tracking params
│   ├── types.py            # Detection / Track / RiskFeature / RiskLevel contracts
│   ├── pipeline.py         # AEBPipeline main chain
│   ├── calibration.py      # per-video self-calibration (joint f·H & horizon estimation)
│   ├── ego_speed.py        # GPS speed from BDD100K info/*.json
│   ├── detectors/          # BaseDetector + YOLOv8 + D-FINE
│   ├── tracker/            # ByteTrack adapter + cython_bbox numpy shim
│   │   └── vendor/bytetrack/   # vendored ByteTrack (MIT, minimal subset, shipped with wheel)
│   ├── distance/           # ground-plane ranging
│   ├── ttc/                # scale TTC / distance TTC / fusion / track history
│   ├── in_path/            # fixed trapezoid ROI (ego-path corridor)
│   └── decision/           # risk state machine
├── examples/
│   ├── quickstart.py       # one-shot demo (--demo / --check)
│   ├── demo_aeb.py         # offline real-video demo
│   └── calibrate_video.py  # per-video self-calibration CLI
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Core principles

### Ranging (ground-plane model)

Assuming a flat ground plane with camera height `H` and focal length `f` (which enter ranging only through their product `f·H`, a scale degeneracy):

$$
D = \frac{f \cdot H}{y_{bottom} - y_{horizon}}
$$

`f·H` and `horizon_y` are calibrated per video (see below), not from global intrinsics.

### Calibration-free scale TTC

For a target's image size `s` (box height or width), its rate of change is related to closing speed, independent of camera calibration:

$$
\tau_{scale} = \frac{s}{\dot{s}}
$$

In implementation, a Theil–Sen robust linear fit is applied to `(t, 1/s)`, yielding slope `b`, so `τ_scale = −1 / (b · s_now)`. Theil–Sen takes the median of pairwise slopes and has a breakdown point of about `29%` (`1 − 1/√2`), robust to a significant fraction of outliers — better than ordinary least squares, whose breakdown point is `0`.

### Distance TTC and fusion

From the ranging sequence `(t, D)`, fit the closing speed `v_close = −slope`, then `τ_distance = D / v_close`. The two sources are fused by taking the more conservative one:

$$
\tau_{fused} = \min(\tau_{scale}, \tau_{distance})
$$

with a `persist_k` frame hold to suppress single-frame jitter.

### Risk state machine

```
NORMAL(0) → ATTENTION(1) → FCW(2) → AEB_WARNING(3)
```

The AEB trigger threshold adapts to ego speed (`t_react` reaction time, `a_max` emergency-braking max deceleration):

$$
\tau_{AEB}(v) = t_{react} + \frac{v}{2 \cdot a_{max}}
$$

FCW / ATTENTION add fixed lead margins on top (defaults `+1.0s` / `+2.0s`). Escalation is immediate; de-escalation has hysteresis to avoid oscillation around thresholds. Required deceleration (**reserved, not currently wired into the state machine**):

$$
a_{req} = \frac{v_{close}^2}{2 \cdot (D - d_{safe})}
$$

Defaults are in `aeb/config.py` (`t_react=0.5`, `a_max=6.86 m/s²`, `d_safe=5.0 m`, `history_len=10`, `persist_k=3`, `fps=30`).

---

## Per-video calibration

BDD100K is crowdsourced, so cameras differ per video and there are no unified intrinsics. This repository provides two calibrations:

1. **Ego speed + scale TTC** (primary entry point): for approximately stationary targets `D = v_ego · τ_scale`, then a Theil–Sen line fit over `(1/D, y_bottom)` jointly solves the slope `f·H` and the intercept `horizon_y` (`aeb/calibration.py: estimate_ground_plane`).
2. **Known target width (reserved)**: `CameraConfig.calibrate_from_known_width()` (currently no call site, kept as a reserved interface) recovers camera height from how pixel width changes with `(y_bottom − horizon)`.

CLI:

```bash
python examples/calibrate_video.py --stem 0571873b-faf718b2
# or
python examples/calibrate_video.py --video /path/to/samples-1k/videos/xxxx.mov --max-frames 200
```

`--stem` mode locates the video directory via the `BDD100K_VIDEO_DIR` environment variable (defaults to the current directory).

---

## Development & testing

Zero-data test suite: no video, detection weights, or network needed — only the core dependencies (numpy / scipy / opencv-python / lap).

```bash
pip install -e ".[dev]"     # or just pip install pytest
pytest -q
```

| Test | Verifies |
| --- | --- |
| `tests/test_ttc_synthetic.py` | synthetic closing scenarios: ranging recovery (median relative error < 10%), dual-source TTC, monotonic four-level state-machine escalation |
| `tests/test_calibration_horizon_synthetic.py` | joint calibration recovers `horizon_y` (truth 320 / 360 / 400, tolerance 8 px) |
| `tests/test_bytetrack_vendor.py` | vendored ByteTrack runs without Cython extensions (numpy shim, numpy≥1.24 alias restoration, real tracked output) |
| `tests/test_import_smoke.py` | import chain, ROI trapezoid derivation, 7-class per-class tracker |
| `tests/test_quickstart_demo.py` | end-to-end `examples/quickstart.py`, asserting the level progression promised in the README |

CI (`.github/workflows/ci.yml`) runs the above tests and the zero-data demo on Ubuntu / Windows × Python 3.10 / 3.12; a separate job runs `pip install -e ".[yolo,dev]"`, validating full dependency resolution including the optional detector backend (yolo extra).

---

## License

This project's code is released under **Apache-2.0**; see [LICENSE](LICENSE).

## Third-party components & acknowledgments

| Component | License | Use |
| --- | --- | --- |
| [ByteTrack](https://github.com/FoundationVision/ByteTrack) | MIT | multi-object tracking (minimal subset vendored at `aeb/tracker/vendor/bytetrack/`, license text shipped with the wheel) |
| [D-FINE](https://github.com/Peterande/D-FINE) | Apache-2.0 | optional high-accuracy detection backend |
| [ultralytics (YOLOv8)](https://github.com/ultralytics/ultralytics) | AGPL-3.0 | optional detection backend (yolo extra) |
| [BDD100K](https://bdd-data.berkeley.edu/) | academic/non-commercial, no redistribution | training data (weights restricted) |

> ⚠️ **License note**: `ultralytics` is AGPL-3.0, which imposes copyleft obligations on derivative works and network-service scenarios. If your distribution needs to avoid those obligations, use the Apache-2.0 D-FINE detector instead (`--detector dfine`). This repository's own code is Apache-2.0.
