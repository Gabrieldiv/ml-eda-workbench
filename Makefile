.DEFAULT_GOAL := help
UV := uv run

.PHONY: help install test lint format notebook report clean

help:
	@echo "Available commands:"
	@echo "  make install    Install all dependencies via uv"
	@echo "  make test       Run the test suite with coverage"
	@echo "  make lint       Check code style and quality with ruff"
	@echo "  make format     Auto-format code with ruff"
	@echo "  make notebook   Launch Jupyter Lab"
	@echo "  make report     Generate a ydata-profiling HTML report (edit path in config)"
	@echo "  make clean      Remove generated files and caches"

install:
	uv sync

test:
	$(UV) pytest

lint:
	$(UV) ruff check .

format:
	$(UV) ruff format .

notebook:
	$(UV) jupyter lab

report:
	$(UV) python -c "\
import pandas as pd; \
from ydata_profiling import ProfileReport; \
from eda_workbench.config import Settings; \
cfg = Settings.from_yaml(); \
df = pd.read_csv(cfg.data.train_path); \
ProfileReport(df, minimal=cfg.profiling.minimal).to_file(cfg.profiling.output_file); \
print('Report saved to', cfg.profiling.output_file)"

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ .coverage coverage.xml htmlcov outputs/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
