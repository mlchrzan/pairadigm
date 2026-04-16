# Contributing to Pairadigm

Thanks for your interest in contributing! This document explains how to get set up, what conventions to follow, and how to submit changes.

---

## Getting Started

### Prerequisites

- Python ≥ 3.10
- `git`

### Installation (development)

```bash
# Clone the repo
git clone https://github.com/mlchrzan/pairadigm.git
cd pairadigm

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install with all development extras
pip install -e ".[dev,test,reward]"
```

The `[dev]` extra pulls in `pytest`, `pytest-cov`, `black`, `ruff`, and `mypy`.

---

## Project Structure

```
pairadigm/
├── pairadigm/
│   ├── __init__.py        # Public API surface
│   ├── core.py            # Pairadigm class (thin orchestrator)
│   ├── client.py          # LLMClient (multi-provider)
│   ├── breakdowns.py      # CGCoT breakdown generation
│   ├── scoring.py         # Bradley-Terry / Davidson model fitting
│   ├── validation.py      # ALT-test, Dawid-Skene EM, IRR, ICC
│   ├── visualization.py   # Plotly & Seaborn plots
│   ├── persistence.py     # Structured save/load (JSON + Parquet)
│   ├── model.py           # Optional: RewardModel (torch)
│   └── _stats.py          # Internal statistics helpers
├── tests/
│   └── test_basic.py      # pytest suite (run without an API key via mocks)
├── pyproject.toml
└── CONTRIBUTING.md        # This file
```

---

## Running Tests

Tests use `pytest` and mock all LLM API calls so **no API key is needed**:

```bash
pytest tests/ -v --tb=short
```

To see coverage:

```bash
pytest tests/ --cov=pairadigm --cov-report=term-missing
```

---

## Code Style

- **Formatter**: `black` (line length 100)
- **Linter**: `ruff` — run `ruff check pairadigm/` before committing
- **Type hints**: Required for all public functions and class methods
- **Docstrings**: NumPy-style for public API, inline comments for implementation details

Auto-format and lint:

```bash
black pairadigm/ tests/
ruff check pairadigm/ --fix
```

---

## Making Changes

### Branching

Create a feature branch off `main`:

```bash
git checkout -b feature/my-feature
```

Keep PRs focused — one logical change per PR makes review faster.

### Commit messages

Use the format: `<type>: <short description>`

Types: `fix`, `feat`, `refactor`, `test`, `docs`, `chore`

Examples:
```
feat: add pair_from_likert() for ordinal prior annotations
fix: correct M-step guard in Dawid-Skene EM (issue #12)
test: add round-trip test for Parquet save/load
```

### API changes

`pairadigm` follows a **deprecation-first** approach. If you rename or remove a public parameter:

1. Keep the old parameter with a `DeprecationWarning` for one minor version.
2. Remove it in the next minor version (document in `CHANGELOG.md`).

See `core.py` for examples of the `llm_decision_col → decision_col` migration.

---

## Adding a New LLM Provider

1. Add a new `_infer_provider` branch in `client.py`.
2. Add a `_generate_<provider>()` method following the existing pattern.
3. Add a `_dispatch()` branch.
4. Update the docstring and the README's provider table.
5. Add a mocked test case in `tests/test_basic.py`.

---

## Submitting a Pull Request

1. Ensure `pytest tests/ -v` passes with no failures.
2. Run `black` and `ruff` to clean up code style.
3. Update `CHANGELOG.md` with a brief description.
4. Open a PR against `main` with a clear title and description.

---

## Reporting Bugs

Please open an issue with:
- Python version and OS
- Pairadigm version (`python -c "import pairadigm; print(pairadigm.__version__)"`)
- Minimal reproducible example
- Full traceback

---

## Code of Conduct

Be kind, constructive, and welcoming. We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
