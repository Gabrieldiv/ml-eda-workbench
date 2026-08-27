"""Unit tests for the Settings dataclasses in config.py."""

from pathlib import Path

from eda_workbench.config import DataConfig, Settings


def test_settings_defaults():
    """Settings() uses sensible defaults when built without a YAML file."""
    cfg = Settings()
    assert cfg.experiment.random_seed == 42
    assert cfg.experiment.test_size == 0.2
    assert cfg.data.target_column == "target"
    assert cfg.bias.report_detail == "summary"


def test_settings_from_yaml(settings):
    """from_yaml() correctly loads values from config/settings.yaml."""
    assert settings.experiment.random_seed == 42
    assert settings.data.path == "data/"
    assert settings.bias.min_group_count == 30
    assert settings.bias.disparate_impact_threshold == 0.8
    assert settings.bias.report_detail == "summary"


def test_data_config_train_path():
    """train_path joins path and train_file correctly."""
    data = DataConfig(path="data/", train_file="train.csv")
    assert data.train_path == Path("data/train.csv")


def test_data_config_test_path():
    """test_path joins path and test_file correctly."""
    data = DataConfig(path="data/", test_file="test.csv")
    assert data.test_path == Path("data/test.csv")


def test_data_config_paths_respect_custom_directory():
    """Paths reflect a non-default data directory."""
    data = DataConfig(path="custom_dir/", train_file="my_train.csv")
    assert data.train_path == Path("custom_dir/my_train.csv")


def test_settings_from_yaml_missing_file_uses_defaults(tmp_path):
    """Falls back to dataclass defaults when the YAML file does not exist."""
    missing_path = tmp_path / "does_not_exist.yaml"
    cfg = Settings.from_yaml(missing_path)

    assert cfg == Settings()
    assert cfg.experiment.random_seed == 42
    assert cfg.data.target_column == "target"
