"""Roadmap of paper techniques by implementation lift.

Canonical inventory of methods from the review paper, tagged Low / Medium /
High / Very high. Identification is Section 6 (plus Section 5 metric families
used as audit instruments). Mitigation is Section 7 and is out of scope for
this workbench.

Human-readable copy: ``docs/bias-roadmap.md``. Keep titles in that file in
sync with the tuples below (``tests/test_bias_roadmap.py`` checks this).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Lift(StrEnum):
    """How hard the technique is to add to this workbench."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very high"


class Status(StrEnum):
    IMPLEMENTED = "implemented"
    NOT_IMPLEMENTED = "not implemented"
    OUT_OF_SCOPE = "out of scope"


LIFT_DEFINITIONS: dict[Lift, str] = {
    Lift.LOW: ("pandas on the profiled table only; no model and no eval bundle."),
    Lift.MEDIUM: (
        "Library stats or metrics on user-supplied tables (predictions, "
        "scores, or features); no training inside the workbench."
    ),
    Lift.HIGH: (
        "Needs a provided model and/or eval features, extra metadata, or "
        "multi-run sensitivity. Still no training inside the workbench."
    ),
    Lift.VERY_HIGH: (
        "Neural internals, generative or production APIs, causal graphs, "
        "or anything that trains a model in this repo."
    ),
}


@dataclass(frozen=True)
class RoadmapItem:
    """One paper technique on the implementation roadmap."""

    id: str
    title: str
    lift: Lift
    status: Status
    paper_section: str
    paper_label: str
    why: str
    module: str | None = None


# ---------------------------------------------------------------------------
# Identification (paper Section 6; metric families from Section 5)
# ---------------------------------------------------------------------------

IDENTIFICATION_ROADMAP: tuple[RoadmapItem, ...] = (
    RoadmapItem(
        id="representational_coverage",
        title="Representational coverage (imbalance, intersectional sparsity)",
        lift=Lift.LOW,
        status=Status.IMPLEMENTED,
        paper_section="6.3",
        paper_label="sec:dataset-and-cohort-bias-auditing",
        why="Group-by counts on the profiled CSV; pandas only.",
        module="eda_workbench.bias.coverage",
    ),
    RoadmapItem(
        id="group_descriptive_stats",
        title="Group descriptive statistics",
        lift=Lift.LOW,
        status=Status.IMPLEMENTED,
        paper_section="6.3",
        paper_label="sec:dataset-and-cohort-bias-auditing",
        why="n, share, and missingness are the same coverage tables.",
        module="eda_workbench.bias.coverage",
    ),
    RoadmapItem(
        id="multi_metric_dashboard",
        title="Multi-metric fairness dashboard on provided eval predictions",
        lift=Lift.MEDIUM,
        status=Status.IMPLEMENTED,
        paper_section="6.1",
        paper_label="sec:metric-based-fairness-auditing",
        why="fairlearn MetricFrame on y_true/y_pred; families in §5.1–5.5.",
        module="eda_workbench.bias.metric_audit",
    ),
    RoadmapItem(
        id="proxy_association_tests",
        title="Proxy association tests (mutual information, chi-square, correlation)",
        lift=Lift.MEDIUM,
        status=Status.NOT_IMPLEMENTED,
        paper_section="6.1",
        paper_label="sec:metric-based-fairness-auditing",
        why="sklearn/scipy tests on the profiled table; no model required.",
    ),
    RoadmapItem(
        id="threshold_free_ks_dprime",
        title="Threshold-free score-distribution tests (Kolmogorov–Smirnov, d′)",
        lift=Lift.MEDIUM,
        status=Status.NOT_IMPLEMENTED,
        paper_section="6.1",
        paper_label="sec:metrics-roc-based",
        why="scipy on provided y_score grouped by protected attribute.",
    ),
    RoadmapItem(
        id="shap_tabular",
        title="SHAP on a provided eval table + model",
        lift=Lift.HIGH,
        status=Status.IMPLEMENTED,
        paper_section="6.2",
        paper_label="sec:explainability-driven-bias-diagnosis",
        why="Needs model + eval_path. Paper names saliency maps, not SHAP.",
        module="eda_workbench.bias.shap_audit",
    ),
    RoadmapItem(
        id="lime",
        title="LIME (local surrogate explanations) on eval rows",
        lift=Lift.HIGH,
        status=Status.NOT_IMPLEMENTED,
        paper_section="6.2",
        paper_label="sec:explainability-driven-bias-diagnosis",
        why="Same eval+model gate as SHAP; extra lime dependency.",
    ),
    RoadmapItem(
        id="acquisition_protocol_stratification",
        title="Acquisition / protocol stratification",
        lift=Lift.HIGH,
        status=Status.NOT_IMPLEMENTED,
        paper_section="6.3",
        paper_label="sec:dataset-and-cohort-bias-auditing",
        why="Needs capture/device/site metadata columns the CSV may not have.",
    ),
    RoadmapItem(
        id="cohort_construction_sensitivity",
        title="Cohort-construction sensitivity",
        lift=Lift.HIGH,
        status=Status.NOT_IMPLEMENTED,
        paper_section="6.3",
        paper_label="sec:dataset-and-cohort-bias-auditing",
        why="Must re-run coverage/metrics under alternate inclusion rules.",
    ),
    RoadmapItem(
        id="bootstrap_cis",
        title="Bootstrap CIs / hypothesis tests on fairness gaps",
        lift=Lift.HIGH,
        status=Status.NOT_IMPLEMENTED,
        paper_section="6.1",
        paper_label="sec:metric-based-fairness-auditing",
        why="Many resamples of the eval predictions; more runtime than MetricFrame.",
    ),
    RoadmapItem(
        id="quality_aware_auditing",
        title="Quality-aware auditing (companion to explanations)",
        lift=Lift.HIGH,
        status=Status.NOT_IMPLEMENTED,
        paper_section="6.2",
        paper_label="sec:explainability-driven-bias-diagnosis",
        why="Needs image-quality or capture-condition scores alongside groups.",
    ),
    RoadmapItem(
        id="dataset_responsibility_screening",
        title="Dataset responsibility / diversity / privacy screening",
        lift=Lift.HIGH,
        status=Status.NOT_IMPLEMENTED,
        paper_section="6.3",
        paper_label="sec:dataset-and-cohort-bias-auditing",
        why="Needs consent, leakage, and regulatory checks beyond column counts.",
    ),
    RoadmapItem(
        id="nuisance_factor_representation",
        title="Nuisance-factor representation beyond demographics",
        lift=Lift.HIGH,
        status=Status.NOT_IMPLEMENTED,
        paper_section="6.3",
        paper_label="sec:dataset-and-cohort-bias-auditing",
        why="Needs task-specific nuisance labels (pose, illumination, traits).",
    ),
    RoadmapItem(
        id="neural_saliency_maps",
        title="Neural saliency / perturbation maps and spatial localization",
        lift=Lift.VERY_HIGH,
        status=Status.NOT_IMPLEMENTED,
        paper_section="6.2",
        paper_label="sec:explainability-driven-bias-diagnosis",
        why="CNN/vision stack, image tensors, and aggregation across images.",
    ),
    RoadmapItem(
        id="intermediate_representation_probes",
        title="Intermediate-representation probes",
        lift=Lift.VERY_HIGH,
        status=Status.NOT_IMPLEMENTED,
        paper_section="6.2",
        paper_label="sec:explainability-driven-bias-diagnosis",
        why="Must hook hidden layers and often train a probe classifier.",
    ),
    RoadmapItem(
        id="generative_prompt_suite_audits",
        title="Generative prompt-suite / output-distribution audits",
        lift=Lift.VERY_HIGH,
        status=Status.NOT_IMPLEMENTED,
        paper_section="6.1",
        paper_label="sec:metric-based-fairness-auditing",
        why="Sample a generative model, estimate attributes, compare to a reference.",
    ),
    RoadmapItem(
        id="production_black_box_apis",
        title="Production black-box API audits",
        lift=Lift.VERY_HIGH,
        status=Status.NOT_IMPLEMENTED,
        paper_section="6.4",
        paper_label="sec:identification-system-level-and-black-box",
        why="Network calls to a deployed system under controlled inputs.",
    ),
    RoadmapItem(
        id="builtin_app_prediction_audits",
        title="Built-in application prediction audits (controlled inputs)",
        lift=Lift.VERY_HIGH,
        status=Status.NOT_IMPLEMENTED,
        paper_section="6.4",
        paper_label="sec:identification-system-level-and-black-box",
        why="Drive a closed app (age/sex estimators) with curated or synthetic sets.",
    ),
    RoadmapItem(
        id="generative_retrieval_portrayal",
        title="Generative / retrieval portrayal audits",
        lift=Lift.VERY_HIGH,
        status=Status.NOT_IMPLEMENTED,
        paper_section="6.4",
        paper_label="sec:identification-system-level-and-black-box",
        why="Prompt image search or T2I, then label portrayal (pose, smile, etc.).",
    ),
    RoadmapItem(
        id="counterfactual_fairness",
        title="Counterfactual fairness",
        lift=Lift.VERY_HIGH,
        status=Status.NOT_IMPLEMENTED,
        paper_section="5.7",
        paper_label="sec:metrics-causal-conditional-constraint-based",
        why="Needs a causal graph and interventions on protected attributes.",
    ),
    RoadmapItem(
        id="context_aware_spatial_analyses",
        title="Context-aware spatial / feature-entanglement analyses",
        lift=Lift.VERY_HIGH,
        status=Status.NOT_IMPLEMENTED,
        paper_section="6.2",
        paper_label="sec:explainability-driven-bias-diagnosis",
        why="Image localization plus ablation of location/background features.",
    ),
)


# ---------------------------------------------------------------------------
# Mitigation (paper Section 7) — out of scope; listed so the inventory is full
# ---------------------------------------------------------------------------

MITIGATION_ROADMAP: tuple[RoadmapItem, ...] = (
    RoadmapItem(
        id="pretrain_reweighting",
        title="Dataset reweighting of underrepresented groups",
        lift=Lift.MEDIUM,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.1",
        paper_label="sec:pre-training",
        why="Table of weights is easy; a trainer must consume them.",
    ),
    RoadmapItem(
        id="pretrain_resampling",
        title="Resampling (over/under-sample, including intersectional groups)",
        lift=Lift.MEDIUM,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.1",
        paper_label="sec:pre-training",
        why="pandas resampling is easy; it still implies retraining.",
    ),
    RoadmapItem(
        id="pretrain_instance_reweighting",
        title="Shortcut / confidence-based instance reweighting",
        lift=Lift.HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.1",
        paper_label="sec:pre-training",
        why="Needs model scores or protected labels to set per-row weights.",
    ),
    RoadmapItem(
        id="pretrain_classical_augmentation",
        title="Classical data augmentation",
        lift=Lift.HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.1",
        paper_label="sec:pre-training",
        why="Image transforms plus a training loop.",
    ),
    RoadmapItem(
        id="pretrain_targeted_acquisition",
        title="Targeted acquisition and curation of sparse cohorts",
        lift=Lift.VERY_HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.1",
        paper_label="sec:pre-training",
        why="Human collection/curation, not a library call.",
    ),
    RoadmapItem(
        id="pretrain_generative_augmentation",
        title="Generative augmentation (GAN / diffusion synthesis)",
        lift=Lift.VERY_HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.1",
        paper_label="sec:pre-training",
        why="Train or call a generator, then audit synthetic bias.",
    ),
    RoadmapItem(
        id="pretrain_staged_training",
        title="Staged training (pretrain then fine-tune on deployment-like cohorts)",
        lift=Lift.VERY_HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.1",
        paper_label="sec:pre-training",
        why="Full training pipeline.",
    ),
    RoadmapItem(
        id="pretrain_domain_adaptation",
        title="Domain adaptation / nuisance-invariant alignment",
        lift=Lift.VERY_HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.1",
        paper_label="sec:pre-training",
        why="Alignment objectives during training.",
    ),
    RoadmapItem(
        id="inproc_loss_margins",
        title="Fairness-aware loss shaping (margins / penalties)",
        lift=Lift.VERY_HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.2",
        paper_label="sec:in-processing-mitigation",
        why="Custom training objective.",
    ),
    RoadmapItem(
        id="inproc_ordinal_task",
        title="Ordinal / hierarchical task reformulation",
        lift=Lift.VERY_HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.2",
        paper_label="sec:in-processing-mitigation",
        why="Changes the learning problem and the trainer.",
    ),
    RoadmapItem(
        id="inproc_adaptive_architecture",
        title="Group-adaptive or coarse-to-fine architecture",
        lift=Lift.VERY_HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.2",
        paper_label="sec:in-processing-mitigation",
        why="Custom model design and training.",
    ),
    RoadmapItem(
        id="inproc_adversarial_removal",
        title="Adversarial removal of sensitive information",
        lift=Lift.VERY_HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.2",
        paper_label="sec:in-processing-mitigation",
        why="Auxiliary adversary plus encoder training.",
    ),
    RoadmapItem(
        id="inproc_information_objectives",
        title="Mutual-information / bottleneck / distance-correlation penalties",
        lift=Lift.VERY_HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.2",
        paper_label="sec:in-processing-mitigation",
        why="Representation-level losses during training.",
    ),
    RoadmapItem(
        id="inproc_conditional_independence",
        title="Conditional-independence losses (equalized odds at representation level)",
        lift=Lift.VERY_HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.2",
        paper_label="sec:in-processing-mitigation",
        why="Fairness constraints inside the trainer.",
    ),
    RoadmapItem(
        id="inproc_contrastive_counterfactual",
        title="Contrastive / counterfactual representation learning",
        lift=Lift.VERY_HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.2",
        paper_label="sec:in-processing-mitigation",
        why="Paired views and a contrastive training loop.",
    ),
    RoadmapItem(
        id="inproc_multimodal_disentanglement",
        title="Multi-modal disentanglement",
        lift=Lift.VERY_HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.2",
        paper_label="sec:in-processing-mitigation",
        why="Multi-encoder training and balanced sampling.",
    ),
    RoadmapItem(
        id="inproc_constraint_solvers",
        title="Constrained / reduction-based fairness solvers during training",
        lift=Lift.VERY_HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.2",
        paper_label="sec:in-processing-mitigation",
        why="In-processing optimization with group constraints.",
    ),
    RoadmapItem(
        id="posttrain_explanation_suppression",
        title="Explanation-guided shortcut suppression",
        lift=Lift.HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.3",
        paper_label="sec:post-training-mitigation",
        why="Uses explanations to mask features; still a mitigation step.",
    ),
    RoadmapItem(
        id="posttrain_feature_refinement_retrain",
        title="Explanation-guided feature refinement and retrain",
        lift=Lift.VERY_HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.3",
        paper_label="sec:post-training-mitigation",
        why="Drop/transform features then train again.",
    ),
    RoadmapItem(
        id="posttrain_group_thresholds",
        title="Group-conditional thresholds / equalized-odds post-processing",
        lift=Lift.HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.3",
        paper_label="sec:post-training-mitigation",
        why="fairlearn post-processors fit a decision rule (training).",
    ),
    RoadmapItem(
        id="posttrain_score_calibration",
        title="Cohort-based score normalization and calibration",
        lift=Lift.HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.3",
        paper_label="sec:post-training-mitigation",
        why="Fits calibration maps; needs a held-out tuning set.",
    ),
    RoadmapItem(
        id="postdeploy_validation_gates",
        title="Deployment validation (non-inferiority, subgroup safety, fairness reporting)",
        lift=Lift.VERY_HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.4",
        paper_label="sec:post-deployment-mitigation",
        why="Protocol, reference workflow, and governance — not EDA.",
    ),
    RoadmapItem(
        id="postdeploy_monitoring",
        title="Continuous monitoring and update governance",
        lift=Lift.VERY_HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.4",
        paper_label="sec:post-deployment-mitigation",
        why="Production ops loop (retrain/recalibrate on live drift).",
    ),
    RoadmapItem(
        id="postdeploy_security_fairness",
        title="Fairness + security evaluation of generative / adversarial pipelines",
        lift=Lift.VERY_HIGH,
        status=Status.OUT_OF_SCOPE,
        paper_section="7.4",
        paper_label="sec:post-deployment-mitigation",
        why="Adversarial testing on deployed or generative systems.",
    ),
)

ROADMAP: tuple[RoadmapItem, ...] = IDENTIFICATION_ROADMAP + MITIGATION_ROADMAP


def items_by_lift(
    items: tuple[RoadmapItem, ...] = IDENTIFICATION_ROADMAP,
) -> dict[Lift, tuple[RoadmapItem, ...]]:
    """Group techniques by implementation lift, preserving declaration order."""
    grouped: dict[Lift, list[RoadmapItem]] = {lift: [] for lift in Lift}
    for item in items:
        grouped[item.lift].append(item)
    return {lift: tuple(group) for lift, group in grouped.items()}
