"""Bias identification (structured results, no Markdown).

Paper: Identification (Section 6, ``sec:identification``), not Mitigation
(Section 7). Technique inventory by implementation lift:
``eda_workbench.bias.roadmap`` and ``docs/bias-roadmap.md``.
"""

from eda_workbench.bias.identify import identify_bias
from eda_workbench.bias.roadmap import (
    IDENTIFICATION_ROADMAP,
    MITIGATION_ROADMAP,
    ROADMAP,
)

__all__ = [
    "IDENTIFICATION_ROADMAP",
    "MITIGATION_ROADMAP",
    "ROADMAP",
    "identify_bias",
]
