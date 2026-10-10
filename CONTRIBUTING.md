# Contributing to mono-aeb

Thanks for your interest. This is a small, focused project: a detector-agnostic monocular
**AEB / FCW decision chain** (detection → tracking → in-path filtering → ranging → TTC → risk
decision). Contributions that keep it small and verifiable are the ones that get merged fastest.

## Ways to contribute

- **Bug report** — use the bug report issue template. A minimal reproduction is worth more than a
  long description.
- **Feature request or design change** — use the feature request template. If it touches a public
  contract (below), describe the motivation before the implementation.
- **Pull request** — small, focused diffs. One concern per PR.
- **Documentation** — `README.md` (English) is the primary document; `README.zh-CN.md` is its
  Chinese counterpart. If you change one, change both.

## Development setup

Requires **Python ≥ 3.10**.

```bash
git clone https://github.com/H12Y10/mono-aeb
cd mono-aeb
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install            # optional, but recommended
pytest -q --cov=aeb --cov-report=term-missing
```

## The zero-data rule

The default test suite must run with **no video, no model weights and no network** — only the core
dependencies (`numpy` / `scipy` / `opencv-python` / `lap`). This is what makes the suite runnable in
CI on every platform and in a container.

- A change that makes `pytest` require a dataset or a detector backend will not be accepted as-is.
- If your work needs weights or video, put it behind an optional extra
  (`[project.optional-dependencies]`) or add it as a benchmark script that skips when the inputs are
  absent — see `benchmarks/` for the pattern.
- Test data is generated in-process (synthetic boxes, synthetic closing trajectories). Prefer that
  over committing fixtures.

## Before opening a pull request

1. `pytest -q` passes.
2. `ruff check .` and `ruff format --check .` pass — or just run `pre-commit run --all-files`.
3. New behaviour has a zero-data test where it can. If existing behaviour changes, say so explicitly
   in the PR description; that is a compatibility statement, not a detail.
4. No model weights and no dataset files are added to the repository.
5. The vendored tracker subset under `aeb/tracker/vendor/bytetrack/` is **not** reformatted — it is
   kept close to upstream so it stays diffable against it.

## Public contracts

Treat these as an API. Changing one is a breaking change and needs a clear description:

| Contract | Location | Notes |
| --- | --- | --- |
| `Detection` / `Track` / `RiskFeature` / `RiskLevel` | `aeb/types.py` | plain dataclasses / `IntEnum` |
| `BaseDetector` | `aeb/detectors/base.py` | the detector-agnostic interface |
| Default thresholds and geometry | `aeb/config.py` | `t_react`, `a_max`, `d_safe`, `history_len`, `persist_k`, `fps`, ego-path trapezoid |
| CLI flags and output text | `examples/*.py` | user-facing; keep output in English |

## Adding a detector backend

1. Subclass `BaseDetector` (`aeb/detectors/base.py`) and implement `detect(frame)`.
2. Return one row per detection as `[x1, y1, x2, y2, score, class]`, in **pixels**, class ids matching
   the 7 AEB classes used by the tracker.
3. Declare the dependency as an optional extra in `pyproject.toml`; the default install must stay
   light. Do not import the heavy dependency at module import time — import it lazily inside the
   class so `--check` keeps working without it.
4. Wire a flag in `examples/demo_aeb.py` and document the backend in both READMEs.
5. Add a zero-data test that exercises the wrapper without the real dependency (a fake detector that
   returns fixed boxes is enough).

## Style

- **User-facing output is English** — CLI text, log lines and exception messages — so the library is
  reusable outside Chinese-language projects. Chinese is fine in comments and in `README.zh-CN.md`.
- Formatting and linting are governed by `ruff` (`pyproject.toml`), line length 100.
- CJK punctuation in comments and docstrings is allowed (`RUF001–RUF003` are disabled for this reason).
- Keep the decision chain dependency-light: no `torch` / `ultralytics` import on the default path.

## Reporting bugs effectively

Include:

- Python version and OS, and how the package was installed (PyPI, `-e .`, container);
- which detector backend was used (`--detector yolo` / `dfine`), or `--check` for the zero-data path;
- the exact command you ran;
- what you expected, what happened instead, and the full traceback;
- for tracking or ranging issues: the input resolution, and whether a per-video calibration was used.

## License

This project is **Apache-2.0**; see [LICENSE](LICENSE). By contributing you agree that your
contribution is licensed under the same terms. The vendored ByteTrack subset remains MIT with its own
license text shipped alongside it (`aeb/tracker/vendor/bytetrack/LICENSE`).
