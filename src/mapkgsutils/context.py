"""Disambiguate ambiguous secondary with context: label, id, and xref evidence.

:class:`ContextSpec` describes a per-row piece of evidence that can
be used to decide which of those two entities a given ambiguous cell actually
means:

* ``label``: an alias/synonym string.
* ``id``    : a related/foreign identifier string.
* ``xref``  : a cross-reference token (e.g. an Ensembl ID) resolved through
  an independent :class:`XrefMapping` crosswalk table.

All three are resolved the same way: confirm *secondary usage* (the evidence
points to the mapping's target), confirm *primary usage* (the evidence points
to the token's own current identity), or leave the cell unresolved. Every
attempt can be recorded as a :class:`DecisionRecord` for an auditable TSV log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mapkgsutils.parsers.base import XrefSource

#: Default for :func:`resolve_ambiguous_with_xref`'s ``trust_unannotated``:
#: an xref record with no ``predicate_id`` is accepted as an equivalence.
DEFAULT_TRUST_UNANNOTATED = True


@dataclass
class XrefRecord:
    """One row of a cross-reference crosswalk table.

    Args:
        subject_id: The cross-reference token, e.g. ``"ENSG00000197471"``.
        object_id: The target primary id, e.g. ``"HGNC:11249"``.
        object_label: The target primary label, e.g. ``"SPN"``.
        predicate_id: The equivalence predicate, e.g. ``"skos:exactMatch"``.
            ``None`` when the source table carries no predicate column.
    """

    subject_id: str
    object_id: str
    object_label: str | None = None
    predicate_id: str | None = None


@dataclass
class XrefMapping:
    """A loaded crosswalk table, indexable by subject (cross-reference) token."""

    records: list[XrefRecord]
    _index: dict[str, list[XrefRecord]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def by_subject(self) -> dict[str, list[XrefRecord]]:
        """Return a ``{subject_id: [records]}`` index, built once and cached."""
        if self._index is None:
            index: dict[str, list[XrefRecord]] = {}
            for record in self.records:
                index.setdefault(record.subject_id, []).append(record)
            self._index = index
        return self._index


def _clean(value: object) -> str | None:
    """Return *value* as a stripped string, or ``None`` for empty/NaN cells."""
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    text = str(value).strip()
    return text or None


def load_xref_mapping(
    path: Path | str,
    *,
    subject_col: str = "subject_id",
    object_col: str = "object_id",
    object_label_col: str = "object_label",
    predicate_col: str = "predicate_id",
    sep: str | None = None,
) -> XrefMapping:
    r"""Load a crosswalk table as an :class:`XrefMapping`.

    Reads either a real SSSOM TSV (a ``#``-prefixed metadata header is
    skipped automatically) or a plain subject/object table.

    Args:
        path: Path to the crosswalk file.
        subject_col: Column with the cross-reference token.
        object_col: Column with the target primary id.
        object_label_col: Column with the target primary label (optional).
        predicate_col: Column with the equivalence predicate (optional).
        sep: Field delimiter.

    Returns:
        An :class:`XrefMapping` with one :class:`XrefRecord` per non-empty
        subject row.
    """
    import pandas as pd

    path = Path(path)
    if sep is None:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep, dtype=str, comment="#")

    records: list[XrefRecord] = []
    for _, row in df.iterrows():
        subject_id = _clean(row.get(subject_col))
        if not subject_id:
            continue
        records.append(
            XrefRecord(
                subject_id=subject_id,
                object_id=_clean(row.get(object_col)) or "",
                object_label=_clean(row.get(object_label_col)),
                predicate_id=_clean(row.get(predicate_col)),
            )
        )
    return XrefMapping(records=records)


def download_xref_source(
    src: XrefSource,
    subject_col: str,
    *,
    show_progress: bool = True,
) -> XrefMapping:
    """Download *src* and build a :class:`XrefMapping` for *subject_col*.

    Args:
        src: A datasource config's suggested crosswalk source (see
            :class:`~mapkgsutils.parsers.base.XrefSource`).
        subject_col: Which column of the downloaded table to use as the
            ``subject_id`` (one of ``src.subject_id_cols.values()``).
        show_progress: Whether to show a download progress bar.

    Returns:
        An :class:`XrefMapping` built from the downloaded table.
    """
    import tempfile

    import pandas as pd

    from mapkgsutils.download import download_file

    with tempfile.TemporaryDirectory() as tmp_dir:
        dest = Path(tmp_dir) / f"{src.id}.{src.format}"
        downloaded = download_file(src.url, dest, show_progress=show_progress)
        sep = "\t" if src.format == "tsv" else ","
        df = pd.read_csv(downloaded, sep=sep, dtype=str)

    if subject_col not in df.columns:
        raise ValueError(f"Column {subject_col!r} not found in downloaded xref source {src.id!r}.")

    records = [
        XrefRecord(
            subject_id=str(row[subject_col]).strip(),
            object_id=str(row.get(src.object_id_col, "") or "").strip(),
            object_label=str(row.get(src.object_label_col, "") or "").strip() or None,
        )
        for _, row in df.iterrows()
        if str(row[subject_col]).strip() not in ("", "nan")
    ]
    return XrefMapping(records=records)


@dataclass
class ContextSpec:
    """One source of per-row evidence used to disambiguate a flagged-ambiguous cell.

    Args:
        kind: ``"label"`` (alias/synonym string, matched via an alias
            index), ``"id"`` (a related identifier string, also matched via
            an alias index), or ``"xref"`` (a cross-reference token, matched
            via an :class:`XrefMapping` crosswalk table).
        column: Name of the DataFrame column carrying this evidence.
        xref_mapping: Required when ``kind == "xref"``.
        predicates: Accepted equivalence predicates for ``kind == "xref"``.
            ``None`` means no restriction (any predicate is accepted).
    """

    kind: Literal["label", "id", "xref"]
    column: str
    xref_mapping: XrefMapping | None = None
    predicates: set[str] | None = None


@dataclass
class DecisionRecord:
    """One disambiguation attempt, for an auditable decision log.

    Args:
        stage: Which context kind made the attempt (``"xref_filter"``,
            ``"label"``, or ``"id"``).
        token: The evidence token that was looked up.
        predicate_id: The candidate's predicate, when applicable.
        candidate: The candidate entity the evidence pointed to.
        accepted: Whether the attempt resolved the ambiguity.
        reason: Human-readable explanation of the outcome.
    """

    stage: str
    token: str
    predicate_id: str | None
    candidate: str | None
    accepted: bool
    reason: str


def write_decision_log(records: list[DecisionRecord], path: Path | str) -> None:
    """Write records to path as a TSV log.

    Columns are ``stage, token, predicate_id, candidate, accepted, reason``.

    Args:
        records: Decision records accumulated during resolution.
        path: Destination TSV path.
    """
    import pandas as pd

    columns = ["stage", "token", "predicate_id", "candidate", "accepted", "reason"]
    df = pd.DataFrame(
        [
            {
                "stage": r.stage,
                "token": r.token,
                "predicate_id": r.predicate_id,
                "candidate": r.candidate,
                "accepted": r.accepted,
                "reason": r.reason,
            }
            for r in records
        ],
        columns=columns,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def resolve_ambiguous_with_xref(
    token: str,
    xref_token: str,
    lkp: dict[str, str],
    xref_index: dict[str, list[XrefRecord]],
    token_to_id: dict[str, str] | None,
    *,
    accepted_predicates: set[str] | None = None,
    trust_unannotated: bool = DEFAULT_TRUST_UNANNOTATED,
) -> tuple[str, str | None, DecisionRecord]:
    """Attempt to resolve an ambiguous token using a cross-reference token.

    Args:
        token: The ambiguous label or ID (appears both as a current primary
            entry and as a secondary one pointing elsewhere).
        xref_token: The row's cross-reference token, e.g. an Ensembl ID.
        lkp: ``{secondary_token: resolved_token}`` lookup.
        xref_index: ``{xref_token: [XrefRecord, ...]}`` (see
            :meth:`XrefMapping.by_subject`).
        token_to_id: ``{primary_token: primary_id}``. ``None`` when *token*
            is already a CURIE (ID mode).
        accepted_predicates: Equivalence predicates to accept. ``None``
            accepts any predicate.
        trust_unannotated: Whether an xref record with no ``predicate_id``
            is accepted as an equivalence.

    Returns:
        A ``(resolved_token, resolved_id, decision)`` tuple. *resolved_token*
        is ``""`` and *resolved_id* is ``None`` when unresolved.
    """
    stage = "xref_filter"
    candidates = xref_index.get(xref_token) or []
    if not candidates:
        return (
            "",
            None,
            DecisionRecord(stage, xref_token, None, None, False, "no crossreference entry"),
        )

    target_token = lkp.get(token, "")

    def _id(tok: str) -> str:
        return (token_to_id or {}).get(tok, tok) if tok else ""

    target_id = _id(target_token)
    own_id = _id(token)

    gate_failure: DecisionRecord | None = None
    mismatch: DecisionRecord | None = None

    for record in candidates:
        candidate_label = record.object_label or record.object_id
        if record.predicate_id is None:
            predicate_ok = trust_unannotated
            reason_ok = "no predicate given, assumed equivalence"
            reason_bad = "no predicate given, trust_unannotated is False"
        else:
            predicate_ok = accepted_predicates is None or record.predicate_id in accepted_predicates
            reason_ok = "predicate accepted"
            reason_bad = f"predicate {record.predicate_id} not in accepted set"

        if not predicate_ok:
            gate_failure = DecisionRecord(
                stage, xref_token, record.predicate_id, candidate_label, False, reason_bad
            )
            continue

        if target_token and (record.object_id == target_id or record.object_label == target_token):
            return (
                target_token,
                target_id or None,
                DecisionRecord(
                    stage, xref_token, record.predicate_id, candidate_label, True, reason_ok
                ),
            )
        if record.object_id == own_id or record.object_label == token:
            return (
                token,
                own_id or None,
                DecisionRecord(
                    stage, xref_token, record.predicate_id, candidate_label, True, reason_ok
                ),
            )

        mismatch = DecisionRecord(
            stage,
            xref_token,
            record.predicate_id,
            candidate_label,
            False,
            "crossreference points to a third entity",
        )

    decision = mismatch or gate_failure
    if decision is None:
        # Unreachable: candidates is non-empty, so the loop above always sets
        # mismatch or gate_failure before falling through to here.
        raise RuntimeError("no decision recorded for a non-empty candidate list")
    return "", None, decision


__all__ = [
    "DEFAULT_TRUST_UNANNOTATED",
    "ContextSpec",
    "DecisionRecord",
    "XrefMapping",
    "XrefRecord",
    "download_xref_source",
    "load_xref_mapping",
    "resolve_ambiguous_with_xref",
    "write_decision_log",
]
