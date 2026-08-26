"""Public API for EDA analysis functions.

Each function lives in its own module (one function per file) for easy
navigation as the analysis suite grows. This file re-exports the public
functions so they remain importable as ``eda_workbench.analysis.<name>``.
"""

from eda_workbench.analysis.analysis_provenance import analysis_provenance
from eda_workbench.analysis.basic_info import basic_info
from eda_workbench.analysis.characteristics_plot import characteristics_plot
from eda_workbench.analysis.duplicated_instances import duplicated_instances
from eda_workbench.analysis.missing_values import missing_values
from eda_workbench.analysis.outliers_instances import outliers_instances
from eda_workbench.analysis.pearson_correlation import pearson_correlation
from eda_workbench.analysis.zero_values import zero_values

__all__ = [
    "basic_info",
    "characteristics_plot",
    "duplicated_instances",
    "missing_values",
    "outliers_instances",
    "pearson_correlation",
    "zero_values",
    "analysis_provenance",
]
