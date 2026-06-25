"""Edge-case tests for mapkgsutils.context (label/id/xref disambiguation)."""

from __future__ import annotations

import pytest

from mapkgsutils.context import XrefRecord, resolve_ambiguous_with_xref


@pytest.mark.parametrize(
    ("xref_object_id", "xref_object_label", "expected"),
    [
        ("HGNC:3", "Y", ("Y", "HGNC:3")),  # xref points to the secondary's target
        ("HGNC:1", "X", ("X", "HGNC:1")),  # xref points to the token's own identity
        ("HGNC:999", "Q", ("", None)),  # xref points to neither: a third entity
    ],
)
def test_xref_resolution_decision_matrix(
    xref_object_id: str, xref_object_label: str, expected: tuple[str, str | None]
) -> None:
    """The three possible xref outcomes for an ambiguous token: target, own identity, or neither."""
    lkp = {"X": "Y"}  # secondary 'X' -> primary 'Y'
    token_to_id = {"X": "HGNC:1", "Y": "HGNC:3"}
    record = XrefRecord(
        subject_id="ENSG1", object_id=xref_object_id, object_label=xref_object_label
    )
    index = {"ENSG1": [record]}
    token, tid, decision = resolve_ambiguous_with_xref("X", "ENSG1", lkp, index, token_to_id)
    assert (token, tid) == expected
    assert decision.accepted == (expected != ("", None))
