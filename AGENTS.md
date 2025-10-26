# Repository Guidelines

## Project Layout
- `bask/` holds the Bayesian sequential optimization library; group new modules beside related functionality (e.g., acquisition strategies in `bask/acquisition`).
- `tests/` mirrors the package structure with `test_*.py` modules; shared fixtures live in `tests/conftest.py`.
- `docs/` stores Sphinx sources; regenerate API pages with `make docs`.
- `examples/` contains runnable notebooks and scripts that demonstrate typical usage paths.
- Root-level tooling (`pyproject.toml`, `noxfile.py`, `Makefile`) configures dependencies, automation, and release packaging.

## Setup & Dev Commands
- Install dependencies through `uv sync`; append `--group docs` when you need the documentation extras.
- Run formatters and linters via `make lint` or `uv run nox -s pre-commit` to execute the full hook suite.
- Execute tests locally with `make test` or `uv run pytest tests` for a focused run.
- Validate against all supported interpreters using `uv run nox -s tests`, which runs pytest across Python 3.10–3.13.
- Build packages with `make dist`; generated artifacts appear in `dist/`.

## Style & Formatting
- Use `black` and `isort` before committing; the shared config keeps formatting deterministic.
- Follow `flake8` rules (80-character lines, Google import order, select=B,C,E,F,W,T4,B9). Resolve warnings instead of adding ignores.
- Prefer `snake_case` for modules and functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Keep docstrings concise and NumPy-style when detailing parameters.

## Testing Expectations
- Name files `test_*.py` and test functions descriptively (e.g., `test_can_sample_posterior`) to mirror the feature under test.
- Lean on pytest fixtures and parametrization to cover corner cases and regression scenarios.
- Measure coverage with `make coverage`; new code should preserve the reported line coverage and add checks for failures you fixed.

## Commits & Pull Requests
- Write focused commits with imperative, sentence-case subjects (`Add posterior sampler`) to match existing history.
- Reference related issues in commit bodies and update `HISTORY.rst` when behavior changes.
- PRs must describe motivation, list validation steps, and confirm `uv run nox -s pre-commit tests` (or equivalent `make` targets) completed successfully; include screenshots for UI-facing artifacts or plots when relevant.
