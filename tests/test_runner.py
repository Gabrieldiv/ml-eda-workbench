"""Unit tests for the EDA runner."""

import os

from eda_workbench.runner import _collect_csv_files, process_folder


def test_collect_csv_files_creates_folder(tmp_path):
    """Creates the data folder when it does not exist."""
    data_dir = str(tmp_path / "data")
    result = _collect_csv_files(data_dir)
    assert os.path.exists(data_dir)
    assert result == []


def test_collect_csv_files_empty_folder(tmp_path):
    """Returns empty list when folder exists but has no CSV files."""
    result = _collect_csv_files(str(tmp_path))
    assert result == []


def test_process_folder_empty_folder(tmp_path):
    """Returns early when no CSV files are found."""
    result = process_folder(data_dir=str(tmp_path))
    assert result is None


def test_process_folder_generates_reports(tmp_path):
    """Processes CSVs and saves reports to output folder."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.csv").write_text("a,b\n1,2\n3,4")

    output_dir = tmp_path / "output"

    process_folder(data_dir=str(data_dir), output_dir=str(output_dir))

    reports = list(output_dir.glob("*.md")) if output_dir.exists() else []
    assert len(reports) > 0

    content = reports[0].read_text(encoding="utf-8")
    assert "## Duplicated Instances" in content
    assert "## Bias Identification" in content
    assert "no protected attributes configured" in content
    assert "sec:identification" not in content


def test_process_folder_skips_plot_when_no_numeric_data(tmp_path):
    """Does not embed an image reference when there's no numeric data to plot."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "text_only.csv").write_text("nome\nAna\nBruno")

    output_dir = tmp_path / "output"

    process_folder(data_dir=str(data_dir), output_dir=str(output_dir))

    reports = list(output_dir.glob("*.md"))
    assert len(reports) > 0

    content = reports[0].read_text(encoding="utf-8")
    assert "![Characteristics Value Ranges]" not in content


def test_process_folder_embeds_violin_plot_when_configured(tmp_path, monkeypatch):
    """Embeds the violin plot image reference in the report when plot_mode
    includes violin."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.csv").write_text(
        "a,b,c\n" + "\n".join(f"{i},{i * 2},{i * 3}" for i in range(1, 40))
    )

    output_dir = tmp_path / "output"

    fake_config = {"correlation": {"plot_mode": "violin", "min_pairwise_n": 2}}
    monkeypatch.setattr("eda_workbench.runner.load_config", lambda: fake_config)

    process_folder(data_dir=str(data_dir), output_dir=str(output_dir))

    reports = list(output_dir.glob("*.md"))
    assert len(reports) > 0

    content = reports[0].read_text(encoding="utf-8")
    assert "![Pearson Correlation Violin Plot]" in content
