# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- In-process CLI tests in `tests/test_cli_module.py`: `aeb.cli.main([...])` is now also driven directly (`--version`, `check`, `demo`, the no-command help, and `demo --out`), alongside the existing subprocess end-to-end cases. The subprocess cases prove the packaged entry points execute; the in-process ones keep the CLI's coverage visible to the in-process report — the suite now reports 68% of `aeb/` overall, 71% over the decision chain excluding the two CLI entry files, and the zero-data case count is 41.

### Fixed

- **Decision-level robustness to far/weak targets** (`aeb/decision/state_machine.py`). Both TTC sources (scale from bbox-growth, distance from range-difference) are dominated by measurement noise when the target is small/far, which produced spurious low TTCs and over-triggered AEB. `decide()` now applies a physical-prior gate: for a same-direction in-path target the implied closing speed (`distance / ttc`) cannot exceed the ego speed by more than 5 m/s (`CLOSING_MARGIN`); inconsistent readings are treated as no-risk and do not participate in escalation or hysteresis. On the 18-segment BDD100K replay this removed five clearly spurious interventions on far/slow/inconsistent targets, moving the intervention rate **88.9% → 61.1%** (16/18 → 11/18) while keeping every genuine near/fast intervention; the ablation `full` AEB-frame count moved **757 → 269** (and `no_inpath` 9655 → 7253). Commit `4923d18`.

## [0.2.1] - 2026-10-12

### Added

- **Console entry points that ship with the wheel** — after `pip install mono-aeb` the package can be run immediately: `mono-aeb demo` / `mono-aeb check` / `mono-aeb run` / `mono-aeb calib`, plus the equivalent `python -m aeb ...` form and `mono-aeb --version`. The implementation moved from `examples/` into the package (`aeb/cli.py`, `aeb/__main__.py`; `examples/` is not part of the wheel, so the zero-data demo could previously only be run from a source checkout), and `[project.scripts]` declares the `mono-aeb` command in `pyproject.toml`.

### Changed

- `examples/quickstart.py`, `examples/demo_aeb.py` and `examples/calibrate_video.py` are now thin wrappers that translate the old flag style into the packaged subcommands, so the documented `python examples/... --demo / --check / ...` invocations keep working unchanged from a checkout. No behaviour change: the demo output is byte-identical, and the zero-data test suite passes unmodified.

## [0.2.0] - 2026-10-11

### Added

- **Road-friction aware risk thresholds (opt-in)**: `RiskThresholds.friction_mu` (default `None`) replaces the fixed `a_max` in the trigger formula `τ_AEB(v) = t_react + v / (2·a_eff)` with a measured `a_eff(μ)` table (`aeb/friction.py`; 10 measured points over `μ ∈ [0.4, 3.5]`, measurement protocol in the module docstring). Values outside the range clamp to the nearest endpoint without extrapolation, and the result is capped at `a_max`. With `friction_mu=None` the behaviour is bit-for-bit identical to the previous fixed-`a_max` formula. Covered by the new zero-data `tests/test_friction_table.py` (endpoints, interpolation, cap, NaN, and the bit-identical default path).
- `CONTRIBUTING.md` — development setup, the zero-data rule for the test suite, the public contracts, how to add a detector backend, and what a useful bug report contains.
- `ROADMAP.md` — what is shipped, what is being considered, what is deliberately out of scope (vehicle control, redistributing weights or data), and the known limitations.
- Issue templates under `.github/ISSUE_TEMPLATE/` — a bug report form, a feature request form, and contact links pointing at the README, the roadmap and the contributing guide.
- `Dockerfile` and `.dockerignore` — `python:3.12-slim` with the `dev` extra. Every core dependency has a manylinux wheel, so the image needs no compiler; the default command runs the zero-data suite. Both READMEs document the container.
- `.gitattributes` — forces LF for `Dockerfile` and `.dockerignore` (a CRLF `.dockerignore` silently stops matching its patterns, pulling bytecode caches into the build context).

### Changed

- Package metadata: the `pyproject.toml` description is now English, matching the English README.
- User-facing text is English throughout: CLI help of all three example scripts, benchmark output under `benchmarks/`, and the runtime error messages of the D-FINE and ByteTrack backends.
- README (both languages): the coverage paragraph enumerates the covered modules and names what actually reports 0% (`dfine_detector.py`, `yolov8_detector.py`, `ego_speed.py`); the development snippet notes that the `--cov` flags need `pytest-cov`; the directory tree lists `tests/`, `benchmarks/`, `Dockerfile` and the community files; the D-FINE section states where its config file is expected to live (`<DFINE_ROOT>/configs/dfine/`).
- README documents installing from PyPI (`pip install mono-aeb`); Chinese documentation punctuation unified, and the warning emoji removed from comments (no behaviour change).
- CI and release workflows: actions upgraded to their current major versions (`checkout@v7`, `setup-python@v7`, `upload-artifact@v7`, `download-artifact@v8`), job and step names in English, and the comments on the UTF-8 environment and the pinned ruff version corrected to match what the jobs do.

### Fixed

- The release workflow now creates the GitHub Release first when a freshly pushed tag does not have one yet, then uploads the sdist and wheel.
- The test table in `README.md` and `README.zh-CN.md` was missing the `tests/test_tracking_synth.py` row (the synthetic tracking benchmark); it is now listed.
- `benchmarks/synth_tracking.py` pointed at a `benchmarks/tracking_eval.py` that does not exist in this repository; the reference is removed and its scope statement now says plainly what the synthetic numbers are and are not.
- `CONTRIBUTING.md` installs `pre-commit` explicitly in the setup snippet (it is not part of the `dev` extra), and no longer points at a benchmark skip-pattern that nothing in the repository demonstrated.
- The `[0.1.2]` coverage note below overstated the range; it is now the same enumeration the README uses.
- Stripped the UTF-8 BOM from two vendored `__init__.py` files.

## [0.1.2] - 2026-10-09

### Added

- `ruff` lint/format configuration in `pyproject.toml` (line length 100; the vendored ByteTrack subset is excluded so it stays diffable against upstream) and a `.pre-commit-config.yaml` hook set. A dedicated CI job runs `ruff check` and `ruff format --check`.
- Coverage reporting: `pytest-cov` added to the `dev` extra and `[tool.coverage.*]` configured; CI prints the per-module table (71% of `aeb/` overall on the zero-data suite — `pipeline.py` 98%, `in_path/fixed_roi.py` 96%, `tracker/bytetrack.py` 96%, `ttc/*` 84–92%, `calibration.py` 85% — while `detectors/*` and `ego_speed.py` need optional backends / real GPS data).
- A `Release` workflow: pushing a `v*` tag builds the sdist and wheel, runs `twine check`, and attaches both to the GitHub Release; publishing to PyPI uses Trusted Publishing and is gated behind a manual `workflow_dispatch` run so a tag push never uploads by itself.

### Changed

- Applied `ruff format` across `aeb/`, `examples/`, `tests/`, `benchmarks/` and the code samples in both READMEs, and fixed the remaining lint findings: import order, `__all__` ordering, unused `noqa` directives, ambiguous `l` variables, `zip(..., strict=...)` made explicit, lambda assignments replaced with `def`, one `%`-format string converted to an f-string. No behaviour change — the zero-data suite and both example entry points are unchanged.

## [0.1.1] - 2026-10-07

### Added
- English `README.md`; the Chinese original is kept at `README.zh-CN.md`, and this `CHANGELOG.md`.

### Changed
- User-facing CLI output (`examples/quickstart.py`, `examples/demo_aeb.py`, `examples/calibrate_video.py`) and library error messages are now English instead of Chinese, for international reuse; the zero-data test suite and the README output samples are updated accordingly.
- Upstream ByteTrack link updated from `ifzhang/ByteTrack` to `FoundationVision/ByteTrack` (the repository moved).
- Package version bumped from `0.1.0` to `0.1.1` (`pyproject.toml`).
- Made `ultralytics` (YOLOv8) an optional extra (`yolo`) instead of a core dependency, so the core package no longer depends on any AGPL-licensed component. (`ac7687e`)

### Fixed
- Removed the in-code `lapx` fallback and replaced it with an install-time replacement: the `lapx` distribution ships a module named `lap`, so `pip install lapx` satisfies `import lap` with no code change. The previous `except ImportError: import lapx as lap` branch could never take effect. (`3cbba89`, `8adcb8b`)

## [0.1.0] - 2026-09-22

### Added
- Initial open-source release of the detector-agnostic monocular AEB/FCW decision chain: detection → tracking → in-path filtering → ranging → TTC → risk decision.
- Calibration-free scale TTC (`τ = s / ṡ`, Theil–Sen robust fit) and speed-adaptive risk thresholds.
- Dual-source TTC fusion (`min` of scale and distance TTC) with `persist_k` frame hold.
- Vendored ByteTrack (MIT, minimal subset) with a pure-numpy `cython_bbox` shim and numpy≥1.24 alias restoration, so inference use needs no C++ toolchain.
- Per-video self-calibration (`f·H` and `horizon_y` joint estimation from GPS ego speed + scale TTC).
- Pluggable detector backends: YOLOv8 (`yolo` extra) and D-FINE (`dfine` extra) behind a common `BaseDetector` interface.
- Zero-data test suite (12 cases) and GitHub Actions CI (Ubuntu / Windows × Python 3.10 / 3.12).

[Unreleased]: https://github.com/H12Y10/mono-aeb/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/H12Y10/mono-aeb/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/H12Y10/mono-aeb/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/H12Y10/mono-aeb/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/H12Y10/mono-aeb/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/H12Y10/mono-aeb/releases/tag/v0.1.0
