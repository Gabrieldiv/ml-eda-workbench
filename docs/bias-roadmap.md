# Bias technique roadmap

Every technique discussed in the review paper, tagged by **how hard it would
be to implement in this workbench**. The same inventory lives in code as
`eda_workbench.bias.roadmap` (`IDENTIFICATION_ROADMAP`, `MITIGATION_ROADMAP`).
Titles here must match that module (`tests/test_bias_roadmap.py` checks).

Section numbers refer to *Bias and Fairness in Intelligent Image Systems*.
LaTeX `\label{}` names are in the Python module.

This workbench implements **identification only** (Section 6). Mitigation
(Section 7) is listed so the paper inventory is complete; those items are
**out of scope**.

## Lift scale

| Lift | Meaning in this repo |
|---|---|
| **Low** | pandas on the profiled table only; no model and no eval bundle. |
| **Medium** | Library stats or metrics on user-supplied tables (predictions, scores, or features); no training inside the workbench. |
| **High** | Needs a provided model and/or eval features, extra metadata, or multi-run sensitivity. Still no training inside the workbench. |
| **Very high** | Neural internals, generative or production APIs, causal graphs, or anything that trains a model in this repo. |

## Identification (Section 6)

Metric *families* used by the dashboard are defined in Section 5
(`sec:metrics`): independence §5.1, separation §5.2, sufficiency §5.3,
ROC §5.4, group-aggregate / worst-case §5.5.

### Low

| Technique | Paper | Status |
|---|---|---|
| Representational coverage (imbalance, intersectional sparsity) | §6.3 `sec:dataset-and-cohort-bias-auditing` | **Implemented** (`eda_workbench.bias.coverage`) |
| Group descriptive statistics | §6.3 | **Implemented** (same coverage tables) |

### Medium

| Technique | Paper | Status |
|---|---|---|
| Multi-metric fairness dashboard on provided eval predictions | §6.1 `sec:metric-based-fairness-auditing` | **Implemented** (`eda_workbench.bias.metric_audit`) |
| Proxy association tests (mutual information, chi-square, correlation) | §6.1 | Not implemented |
| Threshold-free score-distribution tests (Kolmogorov–Smirnov, d′) | §6.1; ROC §5.4 `sec:metrics-roc-based` | Not implemented |

### High

| Technique | Paper | Status |
|---|---|---|
| SHAP on a provided eval table + model | §6.2 family `sec:explainability-driven-bias-diagnosis` (SHAP not named) | **Implemented** (`eda_workbench.bias.shap_audit`) |
| LIME (local surrogate explanations) on eval rows | §6.2 | Not implemented |
| Acquisition / protocol stratification | §6.3 | Not implemented |
| Cohort-construction sensitivity | §6.3 | Not implemented |
| Bootstrap CIs / hypothesis tests on fairness gaps | §6.1 | Not implemented |
| Quality-aware auditing (companion to explanations) | §6.2 | Not implemented |
| Dataset responsibility / diversity / privacy screening | §6.3 | Not implemented |
| Nuisance-factor representation beyond demographics | §6.3 | Not implemented |

### Very high

| Technique | Paper | Status |
|---|---|---|
| Neural saliency / perturbation maps and spatial localization | §6.2 (methods the paper names) | Not implemented |
| Intermediate-representation probes | §6.2; representational metrics §5.6 | Not implemented |
| Generative prompt-suite / output-distribution audits | §6.1 generative | Not implemented |
| Production black-box API audits | §6.4 `sec:identification-system-level-and-black-box` | Not implemented |
| Built-in application prediction audits (controlled inputs) | §6.4 | Not implemented |
| Generative / retrieval portrayal audits | §6.4 | Not implemented |
| Counterfactual fairness | Causal metrics §5.7 `sec:metrics-causal-conditional-constraint-based` | Not implemented |
| Context-aware spatial / feature-entanglement analyses | §6.2 | Not implemented |

## Mitigation (Section 7) — out of scope

None of these will be implemented while the workbench rule is: never train,
never mitigate. Lift still answers “how hard if we ever did.”

### Pre-training (`sec:pre-training`)

| Technique | Lift | Status |
|---|---|---|
| Dataset reweighting of underrepresented groups | Medium | Out of scope |
| Resampling (over/under-sample, including intersectional groups) | Medium | Out of scope |
| Shortcut / confidence-based instance reweighting | High | Out of scope |
| Classical data augmentation | High | Out of scope |
| Targeted acquisition and curation of sparse cohorts | Very high | Out of scope |
| Generative augmentation (GAN / diffusion synthesis) | Very high | Out of scope |
| Staged training (pretrain then fine-tune on deployment-like cohorts) | Very high | Out of scope |
| Domain adaptation / nuisance-invariant alignment | Very high | Out of scope |

### In-processing (`sec:in-processing-mitigation`)

| Technique | Lift | Status |
|---|---|---|
| Fairness-aware loss shaping (margins / penalties) | Very high | Out of scope |
| Ordinal / hierarchical task reformulation | Very high | Out of scope |
| Group-adaptive or coarse-to-fine architecture | Very high | Out of scope |
| Adversarial removal of sensitive information | Very high | Out of scope |
| Mutual-information / bottleneck / distance-correlation penalties | Very high | Out of scope |
| Conditional-independence losses (equalized odds at representation level) | Very high | Out of scope |
| Contrastive / counterfactual representation learning | Very high | Out of scope |
| Multi-modal disentanglement | Very high | Out of scope |
| Constrained / reduction-based fairness solvers during training | Very high | Out of scope |

### Post-training (`sec:post-training-mitigation`)

| Technique | Lift | Status |
|---|---|---|
| Explanation-guided shortcut suppression | High | Out of scope |
| Explanation-guided feature refinement and retrain | Very high | Out of scope |
| Group-conditional thresholds / equalized-odds post-processing | High | Out of scope |
| Cohort-based score normalization and calibration | High | Out of scope |

### Post-deployment (`sec:post-deployment-mitigation`)

| Technique | Lift | Status |
|---|---|---|
| Deployment validation (non-inferiority, subgroup safety, fairness reporting) | Very high | Out of scope |
| Continuous monitoring and update governance | Very high | Out of scope |
| Fairness + security evaluation of generative / adversarial pipelines | Very high | Out of scope |
