"""Guards: never score the profiled CSV; skip when eval artifacts are missing."""

import pandas as pd

from eda_workbench.bias.artifacts import load_eval_bundle
from eda_workbench.bias.identify import _unique_reasons, identify_bias


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


def test_unique_reasons_drops_duplicates_preserving_order():
    assert _unique_reasons(["a", "b", "a", "b", "c"]) == ("a", "b", "c")
    assert _unique_reasons([]) == ()


def test_identify_swallows_metrics_exception_on_valid_bundle(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("fairlearn exploded")

    monkeypatch.setattr("eda_workbench.bias.identify.audit_metrics", boom)
    preds = tmp_path / "preds.csv"
    pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1], "sex": ["f", "m"]}).to_csv(
        preds, index=False
    )
    profiled = pd.DataFrame({"sex": ["f", "m"]})
    profiled_path = tmp_path / "profiled.csv"
    profiled.to_csv(profiled_path, index=False)

    result = identify_bias(
        profiled,
        {
            "protected_attributes": ["sex"],
            "predictions_path": str(preds),
            "separator": ",",
            "encoding": "utf-8",
        },
        profiled_path=str(profiled_path),
    )
    assert result.ran_metrics is False
    skipped = result.metrics.skipped or ""
    assert "Fairness metrics skipped: unexpected error" in skipped
    assert "fairlearn exploded" in skipped


def test_identify_swallows_shap_exception_on_valid_bundle(tmp_path, monkeypatch):
    import joblib
    from sklearn.linear_model import LogisticRegression

    def boom(*args, **kwargs):
        raise RuntimeError("shap exploded")

    monkeypatch.setattr("eda_workbench.bias.identify.explain_eval", boom)
    features = pd.DataFrame({"f1": [0, 1, 0, 1], "f2": [1.0, 2.0, 3.0, 4.0]})
    model = LogisticRegression().fit(features, [0, 1, 0, 1])
    eval_path = tmp_path / "eval.csv"
    eval_frame = features.copy()
    eval_frame["sex"] = ["f", "m", "f", "m"]
    eval_frame.to_csv(eval_path, index=False)
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    profiled = pd.DataFrame({"f1": [9, 9], "sex": ["x", "x"]})
    profiled_path = tmp_path / "profiled.csv"
    profiled.to_csv(profiled_path, index=False)

    result = identify_bias(
        profiled,
        {
            "protected_attributes": ["sex"],
            "eval_path": str(eval_path),
            "model_path": str(model_path),
            "separator": ",",
            "encoding": "utf-8",
        },
        profiled_path=str(profiled_path),
    )
    assert result.ran_shap is False
    skipped = result.shap.skipped or ""
    assert "SHAP skipped: unexpected error" in skipped
    assert "shap exploded" in skipped


def test_load_eval_bundle_missing_and_unreadable_files(tmp_path):
    profiled = tmp_path / "profiled.csv"
    profiled.write_text("a,b\n1,2\n")
    missing = tmp_path / "nope.csv"
    garbage = tmp_path / "garbage.csv"
    garbage.write_bytes(b"\x80\x81 invalid utf-8")
    csv_opts = {"separator": ",", "encoding": "utf-8"}

    preds_missing = load_eval_bundle(
        {"predictions_path": str(missing), **csv_opts},
        profiled_path=str(profiled),
    )
    assert "predictions file not found" in (preds_missing.metrics_skip or "")

    preds_unreadable = load_eval_bundle(
        {"predictions_path": str(garbage), **csv_opts},
        profiled_path=str(profiled),
    )
    assert "could not read predictions" in (preds_unreadable.metrics_skip or "")

    eval_missing = load_eval_bundle(
        {"eval_path": str(missing), **csv_opts},
        profiled_path=str(profiled),
    )
    assert "eval feature table not found" in (eval_missing.shap_skip or "")

    eval_unreadable = load_eval_bundle(
        {"eval_path": str(garbage), **csv_opts},
        profiled_path=str(profiled),
    )
    assert "could not read eval_path" in (eval_unreadable.shap_skip or "")

    eval_ok = tmp_path / "eval.csv"
    eval_ok.write_text("f1,sex\n0,f\n")
    model_missing = load_eval_bundle(
        {
            "eval_path": str(eval_ok),
            "model_path": str(tmp_path / "nope.joblib"),
            **csv_opts,
        },
        profiled_path=str(profiled),
    )
    assert "model file not found" in (model_missing.shap_skip or "")

    bad_model = tmp_path / "bad.joblib"
    bad_model.write_bytes(b"not-a-joblib")
    model_unreadable = load_eval_bundle(
        {
            "eval_path": str(eval_ok),
            "model_path": str(bad_model),
            **csv_opts,
        },
        profiled_path=str(profiled),
    )
    assert "could not load model" in (model_unreadable.shap_skip or "")
