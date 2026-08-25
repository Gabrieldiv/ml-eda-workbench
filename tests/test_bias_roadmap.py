"""The paper-technique roadmap is documented and stays in sync with the docs."""

from pathlib import Path

from eda_workbench.bias.roadmap import (
    IDENTIFICATION_ROADMAP,
    MITIGATION_ROADMAP,
    ROADMAP,
    Status,
    items_by_lift,
)

_DOCS = Path(__file__).resolve().parents[1] / "docs" / "bias-roadmap.md"


def test_roadmap_ids_are_unique():
    ids = [item.id for item in ROADMAP]
    assert len(ids) == len(set(ids))


def test_implemented_identification_has_modules():
    implemented = [
        item for item in IDENTIFICATION_ROADMAP if item.status is Status.IMPLEMENTED
    ]
    assert implemented
    assert all(item.module for item in implemented)


def test_mitigation_is_out_of_scope():
    assert MITIGATION_ROADMAP
    assert all(item.status is Status.OUT_OF_SCOPE for item in MITIGATION_ROADMAP)


def test_identification_covers_all_lifts():
    grouped = items_by_lift(IDENTIFICATION_ROADMAP)
    assert grouped
    for lift, items in grouped.items():
        assert items, f"identification roadmap missing lift {lift}"


def test_docs_list_every_roadmap_title():
    text = _DOCS.read_text(encoding="utf-8")
    missing = [item.title for item in ROADMAP if item.title not in text]
    assert missing == []
