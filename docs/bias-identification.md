# Bias identification

This workbench implements **identification only** (paper Section 6,
`sec:identification`), not mitigation (Section 7). Analysis lives in
`eda_workbench.bias` as structured results; Markdown is rendered separately
by `eda_workbench.reporting`.

Section numbers below refer to *Bias and Fairness in Intelligent Image
Systems: A Systematic Review with Emphasis on Facial Recognition and Facial
Analysis*. LaTeX `\label{}` names are included so the mapping survives
renumbering.

## Paper mapping

| Workbench piece | In the paper? | Where |
|---|---|---|
| Representational coverage (imbalance, intersectional sparsity) | **Yes, by name** | Section 6.3 Dataset and cohort bias auditing (`sec:dataset-and-cohort-bias-auditing`). Quote: “representational coverage audits quantify imbalance, often explicitly at intersectional levels … and visualize sparsity” (citing Mhatre 2025). |
| Group descriptive statistics (n, share, missingness) | **Yes, as the descriptive part of dataset/cohort auditing** | Same §6.3 paragraph on comparing cohorts with “straightforward descriptive statistics”. |
| Multi-metric fairness dashboard | **Yes, by name** | Section 6.1 Metric-based fairness auditing (`sec:metric-based-fairness-auditing`): “run a multi-metric dashboard rather than rely on one fairness definition.” Metric *families* are defined in Section 5 (`sec:metrics`): independence §5.1, separation §5.2, sufficiency §5.3, group-aggregate / worst-case §5.5. Optional per-group AUC is the ROC family §5.4. |
| SHAP attributions | **The identification *family* yes; SHAP is not named** | Section 6.2 Explainability-driven bias diagnosis (`sec:explainability-driven-bias-diagnosis`) asks what the model is using (saliency / perturbation maps, intermediate representations). This workbench instantiates that family for **tabular** sklearn models with `shap.Explainer`. Imaging saliency maps, LIME, and representation probes from §6.2 are not implemented. |

Hard rule: the workbench **never trains a model**, and it **never calls
`predict` / `predict_proba` or SHAP on the profiled dataset CSV**. That file
may have been used for training; we cannot know. Model-dependent analyses run
only on an **explicit evaluation bundle** you supply.

## Evaluation assumptions

Whenever fairness metrics or SHAP run, the report prints an **Assumptions**
block. The workbench cannot verify:

- `eval_path` / `predictions_path` are held-out (or otherwise the set you
  intend to audit), not the training rows.
- `y_pred` / `y_score` were produced by the supplied model (or the system
  being audited) on those same eval rows.
- Protected-attribute labels in the eval bundle are the ones that should be
  used for grouping.
- When both files are given, row alignment between `eval_path` and
  `predictions_path` (by id) is correct.

Representational coverage on the profiled CSV describes **that table's
composition**. It is not a claim about generalization of any model.

## Roadmap

Every paper technique, tagged by implementation lift, is documented in two
places that must stay in sync:

- [docs/bias-roadmap.md](bias-roadmap.md) — the readable list
- `eda_workbench.bias.roadmap` — the canonical inventory in code

The catalog tables that used to live in this file now live there (identification
Section 6, plus mitigation Section 7 as out of scope).

## What is implemented

| Method | Paper | Lift | Status |
|---|---|---|---|
| Representational coverage (counts, shares, imbalance, intersectional sparsity) | §6.3 | Low | Implemented — profiled CSV |
| Group descriptive statistics | §6.3 | Low | Implemented — as part of coverage |
| Multi-metric fairness dashboard (fairlearn `MetricFrame`) | §6.1; families §5.1–5.5 | Medium | Implemented — **provided eval predictions only** |
| Explainability-driven diagnosis via SHAP | §6.2 family (SHAP not named) | High | Implemented — **eval table + model only** |

Not implemented in this pass: proxy association tests, threshold-free KS / d′,
LIME, acquisition/protocol stratification, cohort-construction sensitivity,
bootstrap CIs, neural saliency / representation probes, generative prompt-suite
audits, production black-box APIs, or counterfactual fairness. Mitigation is
out of scope.

## Evaluation bundle

Do **not** join predictions onto the profiled dataset by ID to invent an eval
set. The bundle must be self-contained. Production code will not fall back to
the profiled CSV when `eval_path` is missing, and it will refuse paths that
resolve to the same file as the profiled dataset.

### Predictions CSV (`predictions_path`)

Enough for the fairness dashboard. Easiest export from a training script.

Required:

- `y_true` (or the dataset's configured `target_column`)
- `y_pred` (override with `prediction_column`)
- every configured `protected_attributes` column

Optional:

- `y_score` (override with `score_column`) for per-group AUC
- `id_column` for your own documentation / join to `eval_path` (the workbench
  does not join to the profiled CSV)

If `predictions_path` is missing, metrics are skipped. The workbench will
**not** auto-predict to fill them, even when `eval_path` and `model_path` are
set.

### Eval feature table (`eval_path`)

Required for SHAP, in addition to `model_path`.

- Feature columns the model expects, plus protected attributes (and id if you
  use it to align with predictions)
- Must be a **separately configured path**
- `predict` / SHAP may run on this table only

### Model file (`model_path`)

- `joblib.load` → sklearn-compatible object with `.predict(X)`
- sklearn `Pipeline` is the intended case
- Feature columns: `feature_names_in_` if present, else `model_features` in
  YAML — never “all columns of the profiled CSV”

## Config

Dataset keys (see `titanic_train` in `config/settings.yaml`):

```yaml
protected_attributes: [Sex, Pclass]
eval_path: null           # e.g. data/titanic_eval.csv — NOT the profiled file
predictions_path: null    # e.g. data/titanic_eval_predictions.csv
model_path: null          # e.g. data/titanic_model.joblib
prediction_column: y_pred
score_column: y_score
model_features: null      # used only when the model has no feature_names_in_
```

Global `bias:` keys: `report_detail` (`summary` or `full`), `min_group_count`,
`min_group_share`, `sparse_intersection_share`, `rate_difference_threshold`
(0.1), `disparate_impact_threshold` (0.8), SHAP sample sizes, and
`shap_attribution_gap_threshold`.

`protected_attributes` are **column names in the dataset** (for Titanic,
`Sex` and `Pclass` / ticket class). The workbench does not infer meaning; it
only splits groups by the values in those columns.

If `protected_attributes` is empty or eval artifacts are missing, the affected
section is skipped with an explicit note — the EDA run does not crash.

## What the report shows

The Markdown report does **not** cite paper sections. Default
`bias.report_detail: summary` keeps a short Findings list and one compact
table per block; the rest is in a collapsible `<details>` breakdown.
Set `report_detail: full` to expand every table.

- **Findings** — headlines only (what was flagged, small eval n, expected
  SHAP-on-self). Numbers live in the tables below, not in the bullets.
- **Coverage** — one compact table of group shares on the profiled CSV.
- **Outcomes** — **Actual +** (label prevalence) next to **Predicted +**
  (selection rate), plus TPR and accuracy. Rates are percentages. Eval
  groups smaller than `min_group_count` are called out in Findings.
- **SHAP** — top features by mean abs SHAP. Gaps on the grouping column
  itself are labeled expected when that column is a model feature.
- **Breakdown** — full group tables, assumptions, dashboard, and the
  metric glossary. Precision / accuracy / AUC come from `predictions_path`
  (`y_true` vs `y_pred`; AUC from `y_score`), not the profiled CSV.
