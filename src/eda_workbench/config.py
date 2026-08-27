"""Settings loader for the EDA workbench.

Edit ``config/settings.yaml`` to change experiment parameters, file paths,
plot styles, and profiling options — no code changes needed.

Usage::

    from eda_workbench.config import Settings

    cfg = Settings.from_yaml()  # uses config/settings.yaml by default
    print(cfg.experiment.random_seed)  # 42
    print(cfg.data.target_column)  # "target"
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DataConfig:
    """Paths and column names for the dataset."""

    path: str = "data/"
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    target_column: str = "target"

    @property
    def train_path(self) -> Path:
        return Path(self.path) / self.train_file

    @property
    def test_path(self) -> Path:
        return Path(self.path) / self.test_file


@dataclasses.dataclass
class ExperimentConfig:
    """Reproducibility and splitting parameters."""

    random_seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5


@dataclasses.dataclass
class PlotsConfig:
    """Matplotlib / seaborn rendering defaults."""

    figure_size: list[int] = dataclasses.field(default_factory=lambda: [12, 6])
    style: str = "seaborn-v0_8-whitegrid"
    dpi: int = 150
    save_dir: str = "outputs/figures/"


@dataclasses.dataclass
class ProfilingConfig:
    """ydata-profiling report options."""

    minimal: bool = False
    output_file: str = "outputs/profile_report.html"


@dataclasses.dataclass
class BiasConfig:
    """Thresholds and report verbosity for bias identification.

    Paper: Section 6 identification. Coverage cutoffs follow §6.3
    representational coverage; rate/ratio flags follow the §6.1 dashboard
    (families in §5); SHAP sample sizes support the §6.2 family.
    ``report_detail`` is presentation-only (``summary`` or ``full``).
    """

    report_detail: str = "summary"
    min_group_count: int = 30
    min_group_share: float = 0.05
    sparse_intersection_share: float = 0.01
    rate_difference_threshold: float = 0.1
    disparate_impact_threshold: float = 0.8
    shap_background_size: int = 50
    shap_explain_size: int = 100
    shap_attribution_gap_threshold: float = 0.1


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "config" / "settings.yaml"


@dataclasses.dataclass
class Settings:
    """Top-level project settings.

    Load from YAML::

        cfg = Settings.from_yaml()
        cfg = Settings.from_yaml("path/to/custom.yaml")

    Or use defaults directly::

        cfg = Settings()
    """

    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    experiment: ExperimentConfig = dataclasses.field(default_factory=ExperimentConfig)
    plots: PlotsConfig = dataclasses.field(default_factory=PlotsConfig)
    profiling: ProfilingConfig = dataclasses.field(default_factory=ProfilingConfig)
    bias: BiasConfig = dataclasses.field(default_factory=BiasConfig)

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> Settings:
        """Load settings from a YAML file.

        Args:
            path: Path to the YAML config file. Defaults to
                  ``config/settings.yaml`` relative to the project root.

        Returns:
            A fully populated :class:`Settings` instance.
        """
        config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
        raw: dict[str, Any] = {}
        if config_path.exists():
            with config_path.open() as fh:
                raw = yaml.safe_load(fh) or {}

        return cls(
            data=DataConfig(**raw.get("data", {})),
            experiment=ExperimentConfig(**raw.get("experiment", {})),
            plots=PlotsConfig(**raw.get("plots", {})),
            profiling=ProfilingConfig(**raw.get("profiling", {})),
            bias=BiasConfig(**raw.get("bias", {})),
        )
