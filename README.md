# ML EDA Workbench

A ready-to-use Python environment for **exploratory data analysis**, dataset profiling,
visualizations, feature understanding, and ML-oriented dataset investigation.

---

## Requirements

- [uv](https://docs.astral.sh/uv/) — Python package and virtual environment manager

Install uv (once, on your machine):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/gmichelassi/ml-eda-workbench.git
cd ml-eda-workbench
```

### 2. Install all dependencies

```bash
uv sync
```

This creates a `.venv/` folder and installs every library from the lockfile.
You do **not** need to activate the environment — use `uv run` or the `make` shortcuts below.

### 3. Verify the setup

```bash
make test
```

All smoke tests should pass. If they do, your environment is ready.

---

## Daily Commands

| Task | Command |
|---|---|
| Install / update deps | `make install` |
| Run tests with coverage | `make test` |
| Check code style | `make lint` |
| Auto-format code | `make format` |
| Launch Jupyter Lab | `make notebook` |
| Generate profiling report | `make report` |
| Remove generated files | `make clean` |

You can also call uv directly:

```bash
uv run pytest                   # tests
uv run ruff check .             # lint
uv run ruff format .            # format
uv run jupyter lab              # notebooks
```

---

## Project Structure

```
ml-eda-workbench/
├── config/
│   └── settings.yaml           # ← Edit this to configure your experiment
├── data/                       # Place your CSV / Parquet files here
├── notebooks/
│   └── 01_eda_starter.ipynb    # Starting point for analysis
├── src/
│   └── eda_workbench/
│       ├── __init__.py
│       └── config.py           # Settings loader
├── tests/
│   ├── conftest.py             # Shared fixtures
│   └── test_smoke.py           # Environment health checks
├── pyproject.toml              # All tooling config (ruff, pytest, deps)
└── Makefile                    # Convenience shortcuts
```

---

## Configuration

All experiment parameters live in **`config/settings.yaml`**.
Edit this file to change dataset paths, random seed, plot style, etc.
No code changes are needed.

```yaml
data:
  path: "data/"
  train_file: "train.csv"      # ← point this at your dataset
  target_column: "target"      # ← name of the column you want to predict

experiment:
  random_seed: 42
  test_size: 0.2
  cv_folds: 5

plots:
  style: "seaborn-v0_8-whitegrid"
  dpi: 150
```

Load settings in your code:

```python
from eda_workbench.config import Settings

cfg = Settings.from_yaml()
print(cfg.experiment.random_seed)   # 42
print(cfg.data.train_path)          # PosixPath('data/train.csv')
```

---

## Adding Dependencies

```bash
uv add <package-name>           # runtime dependency
uv add --dev <package-name>     # development-only dependency
```

This updates both `pyproject.toml` and `uv.lock`. Commit both files.

---

## Writing Tests

Tests live in `tests/`. Two fixtures are available out-of-the-box in every test file:

```python
def test_my_function(sample_df, settings):
    # sample_df  → a small 50-row synthetic DataFrame
    # settings   → a Settings object loaded from config/settings.yaml
    assert sample_df.shape[0] == 50
    assert settings.experiment.random_seed == 42
```

Run with coverage:

```bash
make test
```

The XML coverage report is written to `coverage.xml` and is also uploaded as
a GitHub Actions artifact on every CI run.

---

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for both linting and formatting.
Rules enforced:

| Code | Ruleset |
|---|---|
| `E` / `W` | pycodestyle errors and warnings |
| `F` | Pyflakes (unused imports, undefined names) |
| `I` | isort (import ordering) |
| `UP` | pyupgrade (modern Python syntax) |
| `PD` | pandas-vet (common pandas pitfalls) |
| `NPY` | NumPy-specific rules |
| `N` | PEP 8 naming conventions |

CI will fail if `ruff check` or `ruff format --check` do not pass.
Run `make format` before pushing.

---

## Workflow (with Branch Protection)

1. Create a feature branch: `git checkout -b feature/my-analysis`
2. Make your changes, write tests
3. Check everything passes: `make lint && make test`
4. Push and open a Pull Request
5. CI will run automatically — **all checks must be green before merging**
6. Request a review — a mentor approval is required to merge into `main`
