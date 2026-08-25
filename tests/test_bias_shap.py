"""SHAP tests: explain a temp eval frame, never the profiled dataset."""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from eda_workbench.bias.identify import identify_bias
from eda_workbench.bias.shap_audit import explain_eval, resolve_feature_names


def _tiny_estimator():
    features = pd.DataFrame(
        {
            "f1": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "f2": [
                1.0,
                2.0,
                3.0,
                4.0,
                1.5,
                2.5,
                3.5,
                4.5,
                1.2,
                2.2,
                3.2,
                4.2,
                1.8,
                2.8,
                3.8,
                4.8,
            ],
        }
    )
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    model = LogisticRegression().fit(features, labels)
    eval_frame = features.copy()
    eval_frame["sex"] = ["f", "m"] * 8
    return model, eval_frame


def _shap_thresholds() -> dict:
    return {
        "shap_background_size": 8,
        "shap_explain_size": 8,
        "shap_attribution_gap_threshold": 0.5,
    }


def test_shap_explains_eval_frame_created_in_test(tmp_path):
    model, eval_frame = _tiny_estimator()
    model_path = tmp_path / "model.joblib"
    eval_path = tmp_path / "eval.csv"
    joblib.dump(model, model_path)
    eval_frame.to_csv(eval_path, index=False)

    profiled = pd.DataFrame(
        {
            "f1": [99] * 16,
            "f2": [99.0] * 16,
            "sex": ["x"] * 16,
            "trap": range(16),
        }
    )
    profiled_path = tmp_path / "profiled.csv"
    profiled.to_csv(profiled_path, index=False)

    seen_means: list[float] = []
    original = explain_eval

    def tracking_explain(loaded_model, eval_features, *args, **kwargs):
        seen_means.append(float(eval_features["f1"].mean()))
        return original(loaded_model, eval_features, *args, **kwargs)

    import eda_workbench.bias.identify as identify_mod

    identify_mod.explain_eval = tracking_explain
    try:
        result = identify_bias(
            profiled,
            {
                "protected_attributes": ["sex"],
                "eval_path": str(eval_path),
                "model_path": str(model_path),
                "separator": ",",
                "encoding": "utf-8",
            },
            _shap_thresholds(),
            profiled_path=str(profiled_path),
            random_seed=0,
        )
    finally:
        identify_mod.explain_eval = original

    assert result.shap.skipped is None
    assert result.shap.global_importance
    names = {item.feature for item in result.shap.global_importance}
    assert names == {"f1", "f2"}
    assert seen_means
    assert seen_means[0] == eval_frame["f1"].mean()
    assert seen_means[0] != profiled["f1"].mean()


def test_shap_direct_explain_eval_only():
    model, eval_frame = _tiny_estimator()
    result = explain_eval(
        model,
        eval_frame,
        {"protected_attributes": ["sex"]},
        _shap_thresholds(),
        random_seed=0,
    )
    assert result.skipped is None
    assert result.n_explained == 8
    assert result.by_group


def test_resolve_feature_names_from_yaml_when_estimator_has_none():
    class Bare:
        pass

    assert resolve_feature_names(Bare(), {"model_features": ["a", "b"]}) == ["a", "b"]
    assert resolve_feature_names(Bare(), {}) is None


def test_shap_skips_when_eval_missing_model_features():
    model, eval_frame = _tiny_estimator()
    result = explain_eval(
        model,
        eval_frame[["sex"]],
        {"protected_attributes": ["sex"]},
        _shap_thresholds(),
        random_seed=0,
    )
    assert result.skipped is not None
    assert "missing model features" in result.skipped


def test_shap_handles_string_feature_columns():
    """SHAP maskers cannot subtract strings; codes must be decoded for predict."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    features = pd.DataFrame(
        {
            "f1": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "grp": ["a", "b"] * 8,
        }
    )
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    model = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", "passthrough", ["f1"]),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), ["grp"]),
                    ]
                ),
            ),
            ("clf", LogisticRegression()),
        ]
    ).fit(features, labels)
    eval_frame = features.copy()
    eval_frame["sex"] = ["f", "m"] * 8
    result = explain_eval(
        model,
        eval_frame,
        {"protected_attributes": ["sex"]},
        _shap_thresholds(),
        random_seed=0,
    )
    assert result.skipped is None
    assert {item.feature for item in result.global_importance} == {"f1", "grp"}
