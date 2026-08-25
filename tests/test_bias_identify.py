"""Guards: never score the profiled CSV; skip when eval artifacts are missing."""

import pandas as pd

from eda_workbench.bias.identify import identify_bias


def test_missing_eval_path_skips_shap_and_does_not_load_model(tmp_path, monkeypatch):
    def boom_explain(*args, **kwargs):
        raise AssertionError("explain_eval must not run without eval_path")

    def boom_load(*args, **kwargs):
        raise AssertionError("joblib.load must not run when eval_path is missing")

    monkeypatch.setattr("eda_workbench.bias.identify.explain_eval", boom_explain)
    monkeypatch.setattr("eda_workbench.bias.artifacts.joblib.load", boom_load)

    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"not-a-model")
    profiled = pd.DataFrame({"f1": [1, 0], "f2": [1.0, 2.0], "sex": ["f", "m"]})
    profiled_path = tmp_path / "profiled.csv"
    profiled.to_csv(profiled_path, index=False)

    result = identify_bias(
        profiled,
        {
            "protected_attributes": ["sex"],
            "model_path": str(model_path),
            "eval_path": None,
        },
        profiled_path=str(profiled_path),
    )
    assert result.shap.skipped is not None
    assert "eval_path" in result.shap.skipped
    assert result.ran_shap is False
    assert result.metrics.skipped is not None
    assert "predictions_path" in result.metrics.skipped


def test_eval_path_matching_profiled_csv_skips_shap(tmp_path, monkeypatch):
    def boom_explain(*args, **kwargs):
        raise AssertionError("explain_eval must not run on the profiled dataset")

    def boom_load(*args, **kwargs):
        raise AssertionError(
            "joblib.load must not run when eval_path is the profiled CSV"
        )

    monkeypatch.setattr("eda_workbench.bias.identify.explain_eval", boom_explain)
    monkeypatch.setattr("eda_workbench.bias.artifacts.joblib.load", boom_load)

    profiled = pd.DataFrame({"f1": [1, 0], "f2": [1.0, 2.0], "sex": ["f", "m"]})
    profiled_path = tmp_path / "profiled.csv"
    profiled.to_csv(profiled_path, index=False)
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"not-a-model")

    result = identify_bias(
        profiled,
        {
            "protected_attributes": ["sex"],
            "eval_path": str(profiled_path),
            "model_path": str(model_path),
            "separator": ",",
            "encoding": "utf-8",
        },
        profiled_path=str(profiled_path),
    )
    assert result.shap.skipped is not None
    assert "profiled" in result.shap.skipped.lower()


def test_eval_without_predictions_does_not_auto_predict(tmp_path, monkeypatch):
    """SHAP may use eval_path; metrics must not be filled by scoring."""

    def boom_metrics(*args, **kwargs):
        raise AssertionError("audit_metrics must not run without predictions_path")

    monkeypatch.setattr("eda_workbench.bias.identify.audit_metrics", boom_metrics)

    eval_path = tmp_path / "eval.csv"
    pd.DataFrame({"f1": [0, 1], "f2": [1.0, 2.0], "sex": ["f", "m"]}).to_csv(
        eval_path, index=False
    )
    profiled = pd.DataFrame({"f1": [9, 9], "sex": ["x", "x"]})
    profiled_path = tmp_path / "profiled.csv"
    profiled.to_csv(profiled_path, index=False)

    result = identify_bias(
        profiled,
        {
            "protected_attributes": ["sex"],
            "eval_path": str(eval_path),
            "predictions_path": None,
            "model_path": None,
            "separator": ",",
            "encoding": "utf-8",
        },
        profiled_path=str(profiled_path),
    )
    assert result.ran_metrics is False
    assert "predictions_path" in (result.metrics.skipped or "")
    assert result.shap.skipped is not None
    assert "model_path" in result.shap.skipped


def test_identify_coverage_still_runs_without_eval_artifacts():
    frame = pd.DataFrame({"sex": ["f"] * 8 + ["m"] * 2})
    result = identify_bias(
        frame,
        {"protected_attributes": ["sex"]},
        {"min_group_count": 5, "min_group_share": 0.2},
    )
    assert result.coverage.skipped is None
    assert result.coverage.attributes
    assert result.ran_metrics is False
    assert result.ran_shap is False


def test_predictions_matching_profiled_csv_skips_metrics(tmp_path):
    profiled = pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1], "sex": ["f", "m"]})
    profiled_path = tmp_path / "profiled.csv"
    profiled.to_csv(profiled_path, index=False)

    result = identify_bias(
        profiled,
        {
            "protected_attributes": ["sex"],
            "predictions_path": str(profiled_path),
            "separator": ",",
            "encoding": "utf-8",
        },
        profiled_path=str(profiled_path),
    )
    assert result.ran_metrics is False
    assert "profiled" in (result.metrics.skipped or "").lower()


def test_missing_eval_path_does_not_call_predict_on_profiled(monkeypatch, tmp_path):
    class Spy:
        def __init__(self):
            self.calls = 0

        def predict(self, features):
            self.calls += 1
            return [0] * len(features)

        def predict_proba(self, features):
            self.calls += 1
            return [[0.5, 0.5]] * len(features)

    spy = Spy()
    monkeypatch.setattr("eda_workbench.bias.artifacts.joblib.load", lambda path: spy)
    profiled = pd.DataFrame({"f1": [1, 0], "sex": ["f", "m"]})
    result = identify_bias(
        profiled,
        {
            "protected_attributes": ["sex"],
            "model_path": str(tmp_path / "unused.joblib"),
        },
        profiled_path=str(tmp_path / "profiled.csv"),
    )
    assert spy.calls == 0
    assert result.ran_shap is False
    assert "eval_path" in (result.shap.skipped or "")
