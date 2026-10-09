# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ruff` lint/format configuration in `pyproject.toml` (line length 100; the vendored ByteTrack subset is excluded so it stays diffable against upstream) and a `.pre-commit-config.yaml` hook set. A dedicated CI job runs `ruff check` and `ruff format --check`.
- Coverage reporting: `pytest-cov` added to the `dev` extra and `[tool.coverage.*]` configured; CI prints the per-module table (71% of `aeb/` overall on the zero-data suite — the decision chain itself is 84–98%, while `detectors/*` and `ego_speed.py` need optional backends / real GPS data).
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

[Unreleased]: https://github.com/H12Y10/mono-aeb/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/H12Y10/mono-aeb/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/H12Y10/mono-aeb/releases/tag/v0.1.0
