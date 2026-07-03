"""Context-aided resolver for identifier and label reconciliation.

A mapping set is a set of directed ``subject --predicate--> object`` edges. Any
reconciliation problem reduces to the same operation: map each input token along
its edge to the object.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal, Union, overload

from mapkgsutils.context import (
    ContextSpec,
    DecisionRecord,
    XrefMapping,
    XrefRecord,
    resolve_ambiguous_with_xref,
    write_decision_log,
)
from mapkgsutils.logging import logger

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from mapkgsutils.parsers.base import BaseMappingSet

On = Literal["id", "label"]
Values = Union[str, list[str], "pd.DataFrame"]

_SEP = re.compile(r"[|,;\s]+")

_FIELDS: dict[On, tuple[str, str, str]] = {
    "id": ("subject_id", "object_id", "_primary_ids"),
    "label": ("subject_label", "object_label", "_primary_labels"),
}


def build_lookup(mapping_set: BaseMappingSet, *, on: On = "id") -> dict[str, str]:
    """Return ``{subject: object}`` over *on* (``"id"`` or ``"label"``) fields."""
    subject_field, object_field, _ = _FIELDS[on]
    lookup: dict[str, str] = {}
    for m in mapping_set.mappings or []:
        subject = str(getattr(m, subject_field, None) or "")
        obj = str(getattr(m, object_field, None) or "")
        if subject:
            lookup[subject] = obj
    return lookup


def build_ambiguous(mapping_set: BaseMappingSet, *, on: On = "id") -> set[str]:
    """Return tokens that are both a mapping subject and a mapping target over *on* fields.

    Targets come from the stored target set (``_primary_ids``/``_primary_labels``)
    when present, else from the mappings' object values.
    """
    subject_field, object_field, primary_attr = _FIELDS[on]
    stored = (
        object.__getattribute__(mapping_set, primary_attr)
        if hasattr(mapping_set, primary_attr)
        else set()
    )
    primary_tokens: set[str] = (set(stored.keys()) if isinstance(stored, dict) else stored) or {
        str(getattr(m, object_field, None) or "") for m in (mapping_set.mappings or [])
    } - {""}
    subject_tokens: set[str] = {
        str(getattr(m, subject_field, None) or "") for m in (mapping_set.mappings or [])
    } - {""}
    return subject_tokens & primary_tokens


def build_alias_index(
    mapping_set: BaseMappingSet,
    *,
    exclude_predicates: set[str] | None = None,
) -> dict[str, list[str]]:
    """Return ``{object_id: [subject_label]}`` of alias evidence.

    Each retained edge asserts that ``subject_label`` names entity ``object_id``.
    Edges whose predicate is in *exclude_predicates* carry the relation under
    resolution, not identity evidence, and are skipped.
    """
    excluded = exclude_predicates or set()
    index: dict[str, list[str]] = {}
    for m in mapping_set.mappings or []:
        if str(getattr(m, "predicate_id", None) or "") in excluded:
            continue
        obj_id = str(getattr(m, "object_id", None) or "")
        subject_label = str(getattr(m, "subject_label", None) or "")
        if obj_id and subject_label:
            index.setdefault(obj_id, []).append(subject_label)
    return index


def build_primary_token_to_id(mapping_set: BaseMappingSet) -> dict[str, str]:
    """Return ``{object_label: object_id}`` from mappings and stored target labels."""
    result: dict[str, str] = {}
    for m in mapping_set.mappings or []:
        label = str(getattr(m, "object_label", None) or "")
        oid = str(getattr(m, "object_id", None) or "")
        if label and oid:
            result[label] = oid
    stored: dict[str, set[str]] | None = getattr(mapping_set, "_primary_labels", None)
    if isinstance(stored, dict):
        for label, ids in stored.items():
            if label and ids and label not in result:
                result[label] = next(iter(ids))
    return result


def resolve_ambiguous_with_hints(
    ambiguous_token: str,
    user_aliases: list[str],
    lkp: dict[str, str],
    alias_index: dict[str, list[str]],
    token_to_id: dict[str, str] | None = None,
) -> tuple[str, str | None]:
    """Resolve an ambiguous token using user-supplied alias hints.

    Returns ``(target_token, target_id)`` when a hint matches the mapping
    target's identity or aliases; ``(ambiguous_token, own_id)`` when a hint
    matches the token's own aliases; ``("", None)`` otherwise.

    Args:
        ambiguous_token: Token that is both a mapping subject and a mapping target.
        user_aliases: Alias strings supplied for the row.
        lkp: ``{subject: object}`` lookup.
        alias_index: ``{object_id: [alias]}`` from :func:`build_alias_index`.
        token_to_id: ``{token: object_id}``. ``None`` treats each token as its
            own id.
    """
    if not user_aliases:
        return ("", None)

    target_token = lkp.get(ambiguous_token, "")

    def _id(tok: str) -> str:
        return (token_to_id or {}).get(tok, tok)

    target_id = _id(target_token) if target_token else ""
    user_set = set(user_aliases)

    if target_token and (
        target_token in user_set
        or (target_id and target_id in user_set)
        or (target_id and user_set & set(alias_index.get(target_id, [])))
    ):
        return (target_token, target_id if target_id else None)

    own_id = _id(ambiguous_token)
    if own_id and user_set & set(alias_index.get(own_id, [])):
        return (ambiguous_token, own_id if own_id else None)

    return ("", None)


def _warn_ambiguous(ambiguous_found: set[str], kind: str) -> None:
    """Log the tokens left blank because their resolution was ambiguous."""
    if not ambiguous_found:
        return
    listed = ", ".join(sorted(ambiguous_found))
    logger.warning(
        "%d ambiguous %s(s) were left blank because the same %s is both a mapping "
        "subject and a mapping target. Provide alias or cross-reference context, or "
        "resolve manually. Ambiguous %s(s): %s",
        len(ambiguous_found),
        kind,
        kind.lower(),
        kind,
        listed,
    )


def _split_ids(raw: str) -> list[str]:
    """Split *raw* on ``|``, ``,``, ``;``, or whitespace, dropping empty tokens."""
    return [tok for tok in _SEP.split(raw.strip()) if tok]


def _resolve_tokens(
    tokens: list[str],
    lookup: dict[str, str],
    ambiguous: set[str],
    ambiguous_found: set[str],
) -> list[str]:
    """Map each token to its object, blank for ambiguous, itself for unknown."""
    result: list[str] = []
    for tok in tokens:
        if tok in ambiguous:
            ambiguous_found.add(tok)
            result.append("")
        else:
            result.append(lookup.get(tok, tok))
    return result


def _resolve_string(
    raw: str,
    lookup: dict[str, str],
    ambiguous: set[str],
    ambiguous_found: set[str],
) -> str:
    """Resolve every token in *raw*, rejoined on its first separator."""
    sep_match = re.search(r"[|,;\s]", raw)
    sep = sep_match.group(0) if sep_match else ""
    tokens = _split_ids(raw)
    resolved = _resolve_tokens(tokens, lookup, ambiguous, ambiguous_found)
    return sep.join(resolved)


def _update_str(ids: str, lkp: dict[str, str], amb: set[str], kind: str) -> dict[str, str]:
    """Resolve a delimited string into ``{token: resolved}``."""
    ambiguous_found: set[str] = set()
    result: dict[str, str] = {}
    for tok in dict.fromkeys(_split_ids(ids)):
        if tok in amb:
            ambiguous_found.add(tok)
            result[tok] = ""
        else:
            result[tok] = lkp.get(tok, tok)
    _warn_ambiguous(ambiguous_found, kind)
    return result


def _update_list(ids: list[str], lkp: dict[str, str], amb: set[str], kind: str) -> dict[str, str]:
    """Resolve a list of delimited strings into ``{token: resolved}``."""
    unique: dict[str, None] = {}
    for item in ids:
        for tok in _split_ids(item):
            unique[tok] = None
    ambiguous_found: set[str] = set()
    result: dict[str, str] = {}
    for tok in unique:
        if tok in amb:
            ambiguous_found.add(tok)
            result[tok] = ""
        else:
            result[tok] = lkp.get(tok, tok)
    _warn_ambiguous(ambiguous_found, kind)
    return result


def _resolve_columns(at: str | list[str] | None, df: pd.DataFrame, col_label: str) -> list[str]:
    """Validate *at* against *df* and return the target column list."""
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{col_label!r} must be a str, list[str], or pandas.DataFrame, "
            f"got {type(df).__name__!r}."
        )
    if at is None:
        raise ValueError(f"When {col_label!r} is a DataFrame you must specify 'at'.")
    columns = [at] if isinstance(at, str) else list(at)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Column(s) not found in DataFrame: {missing}")
    return columns


def _update_dataframe(
    df: pd.DataFrame,
    at: str | list[str] | None,
    suffix: str,
    lkp: dict[str, str],
    amb: set[str],
    kind: str,
    col_label: str,
) -> pd.DataFrame:
    """Add a resolved ``<col><suffix>`` column for each column in *at*."""
    columns = _resolve_columns(at, df, col_label)
    ambiguous_found: set[str] = set()
    result = df.copy()
    for col in columns:
        result[col + suffix] = (
            result[col]
            .astype(str)
            .map(
                lambda cell, _lkp=lkp, _amb=amb, _af=ambiguous_found: _resolve_string(
                    str(cell), _lkp, _amb, _af
                )
            )
        )
    _warn_ambiguous(ambiguous_found, kind)
    return result


def _raw_context_value(df: pd.DataFrame, idx: int, column: str) -> str:
    """Return the row's evidence value for *column*, or ``""`` when absent/NaN."""
    if column not in df.columns:
        return ""
    raw = str(df.iloc[idx][column])
    return raw if raw.strip() not in {"", "nan"} else ""


def _hint_decision(
    ambiguous_token: str,
    user_aliases: list[str],
    lkp: dict[str, str],
    alias_index: dict[str, list[str]],
    token_to_id: dict[str, str] | None,
    stage: str,
) -> tuple[str, str | None, DecisionRecord]:
    """Resolve via :func:`resolve_ambiguous_with_hints`, returning a decision record."""
    token = "|".join(user_aliases)
    resolved_tok, resolved_id = resolve_ambiguous_with_hints(
        ambiguous_token, user_aliases, lkp, alias_index, token_to_id
    )
    if not resolved_tok:
        reason = (
            "no alias hints given" if not user_aliases else "no alias hint matched either identity"
        )
        return "", None, DecisionRecord(stage, token, None, None, False, reason)
    reason = (
        "alias hint matched mapping target"
        if resolved_tok != ambiguous_token
        else "alias hint matched own identity"
    )
    return resolved_tok, resolved_id, DecisionRecord(stage, token, None, resolved_tok, True, reason)


def _resolve_cell_with_context(
    cell: str,
    lkp: dict[str, str],
    amb: set[str],
    specs: list[ContextSpec],
    raw_values: list[str],
    xref_indices: list[dict[str, list[XrefRecord]] | None],
    alias_index: dict[str, list[str]],
    token_to_id: dict[str, str] | None,
    ambiguous_found: set[str],
    decisions: list[DecisionRecord],
) -> tuple[str, str | None]:
    """Resolve one cell, trying each spec in order on ambiguous tokens.

    The first :class:`~mapkgsutils.context.ContextSpec` that resolves a token
    wins; every attempt is appended to *decisions*. Returns
    ``(resolved_value, resolved_id)``.
    """
    sep_match = re.search(r"[|,;\s]", cell)
    sep = sep_match.group(0) if sep_match else ""
    tokens = _split_ids(cell)
    resolved: list[str] = []
    resolved_id: str | None = None
    for tok in tokens:
        if tok in amb:
            hint_tok = ""
            hint_id: str | None = None
            for spec, raw_val, xref_index in zip(specs, raw_values, xref_indices, strict=True):
                if spec.kind == "xref":
                    if xref_index is None:
                        continue
                    hint_tok, hint_id, decision = resolve_ambiguous_with_xref(
                        tok,
                        raw_val,
                        lkp,
                        xref_index,
                        token_to_id,
                        accepted_predicates=spec.predicates,
                    )
                else:
                    user_aliases = _split_ids(raw_val) if raw_val else []
                    hint_tok, hint_id, decision = _hint_decision(
                        tok, user_aliases, lkp, alias_index, token_to_id, stage=spec.kind
                    )
                decisions.append(decision)
                if hint_tok:
                    break
            if hint_tok:
                resolved.append(hint_tok)
                resolved_id = hint_id
            else:
                ambiguous_found.add(tok)
                resolved.append("")
        else:
            resolved_tok = lkp.get(tok, tok)
            resolved.append(resolved_tok)
            if resolved_id is None:
                resolved_id = (token_to_id or {}).get(resolved_tok) or resolved_tok
    return sep.join(resolved), resolved_id


def _update_dataframe_with_context(
    df: pd.DataFrame,
    at: str | list[str] | None,
    suffix: str,
    lkp: dict[str, str],
    amb: set[str],
    kind: str,
    col_label: str,
    specs: list[ContextSpec],
    alias_index: dict[str, list[str]],
    token_to_id: dict[str, str] | None,
    report_path: Path | str | None,
) -> pd.DataFrame:
    """Add resolved ``<col><suffix>`` and ``<col><suffix>_id`` columns using *specs*.

    Each :class:`~mapkgsutils.context.ContextSpec` names a column of per-row
    evidence. With *report_path*, every attempt is written there as a TSV log.
    """
    columns = _resolve_columns(at, df, col_label)
    xref_indices: list[dict[str, list[XrefRecord]] | None] = [
        spec.xref_mapping.by_subject() if spec.kind == "xref" and spec.xref_mapping else None
        for spec in specs
    ]

    result = df.copy()
    decisions: list[DecisionRecord] = []
    for col in columns:
        ambiguous_found: set[str] = set()
        new_values: list[str] = []
        new_ids: list[str | None] = []
        for idx in range(len(result)):
            cell = str(result.iloc[idx][col])
            raw_values = [_raw_context_value(result, idx, spec.column) for spec in specs]
            val, rid = _resolve_cell_with_context(
                cell,
                lkp,
                amb,
                specs,
                raw_values,
                xref_indices,
                alias_index,
                token_to_id,
                ambiguous_found,
                decisions,
            )
            new_values.append(val)
            new_ids.append(rid)
        result[col + suffix] = new_values
        result[col + suffix + "_id"] = new_ids
        _warn_ambiguous(ambiguous_found, kind)

    if report_path is not None:
        write_decision_log(decisions, report_path)

    return result


def _build_context_specs(
    synonyms: str | list[str] | None,
    xref: str | None,
    xref_mapping: XrefMapping | None,
    xref_predicates: set[str] | None,
    context: ContextSpec | list[ContextSpec] | None,
) -> list[ContextSpec]:
    """Assemble ``synonyms``/``xref``/``context`` kwargs into ordered specs."""
    specs: list[ContextSpec] = []
    if isinstance(synonyms, str):
        specs.append(ContextSpec(kind="label", column=synonyms))
    if xref is not None:
        if xref_mapping is None:
            raise ValueError("'xref_mapping' is required when 'xref' is given.")
        specs.append(
            ContextSpec(
                kind="xref", column=xref, xref_mapping=xref_mapping, predicates=xref_predicates
            )
        )
    if context is not None:
        specs.extend([context] if isinstance(context, ContextSpec) else list(context))
    return specs


@overload
def resolve(
    values: str,
    mapping_set: BaseMappingSet,
    *,
    on: On = ...,
    at: None = ...,
    suffix: str = ...,
    lookup: dict[str, str] | None = ...,
    ambiguous: set[str] | None = ...,
) -> dict[str, str]: ...


@overload
def resolve(
    values: list[str],
    mapping_set: BaseMappingSet,
    *,
    on: On = ...,
    at: None = ...,
    suffix: str = ...,
    lookup: dict[str, str] | None = ...,
    ambiguous: set[str] | None = ...,
) -> dict[str, str]: ...


@overload
def resolve(
    values: pd.DataFrame,
    mapping_set: BaseMappingSet,
    *,
    on: On = ...,
    at: str | list[str],
    suffix: str = ...,
    lookup: dict[str, str] | None = ...,
    ambiguous: set[str] | None = ...,
    synonyms: str | list[str] | None = ...,
    alias_mapping_set: BaseMappingSet | None = ...,
    xref: str | None = ...,
    xref_mapping: XrefMapping | None = ...,
    xref_predicates: set[str] | None = ...,
    relation_predicates: set[str] | None = ...,
    report_path: Path | str | None = ...,
    context: ContextSpec | list[ContextSpec] | None = ...,
) -> pd.DataFrame: ...


def resolve(
    values: Values,
    mapping_set: BaseMappingSet,
    *,
    on: On = "id",
    at: str | list[str] | None = None,
    suffix: str = "_resolved",
    lookup: dict[str, str] | None = None,
    ambiguous: set[str] | None = None,
    synonyms: str | list[str] | None = None,
    alias_mapping_set: BaseMappingSet | None = None,
    xref: str | None = None,
    xref_mapping: XrefMapping | None = None,
    xref_predicates: set[str] | None = None,
    relation_predicates: set[str] | None = None,
    report_path: Path | str | None = None,
    context: ContextSpec | list[ContextSpec] | None = None,
) -> dict[str, str] | pd.DataFrame:
    """Resolve *values* through *mapping_set* over *on* (``"id"`` or ``"label"``).

    *values* is a delimited string, a list of such strings, or a DataFrame (with
    *at* naming the column(s) to resolve). Unknown tokens are kept; ambiguous
    tokens are blanked unless *synonyms*, *xref*, or *context* evidence resolves
    them, in which case a companion ``<col><suffix>_id`` column is added.

    *relation_predicates* name the edges carrying the relation under resolution;
    they are excluded from the alias index built for hint resolution. For
    ``on="id"`` that index comes from *alias_mapping_set*; for ``on="label"`` it
    comes from *mapping_set*.
    """
    lkp = lookup if lookup is not None else build_lookup(mapping_set, on=on)
    amb = ambiguous if ambiguous is not None else build_ambiguous(mapping_set, on=on)
    kind = "ID" if on == "id" else "label"
    col_label = "values"

    if isinstance(values, str):
        return _update_str(values, lkp, amb, kind)
    if isinstance(values, list):
        return _update_list(values, lkp, amb, kind)

    specs = _build_context_specs(synonyms, xref, xref_mapping, xref_predicates, context)
    if not specs:
        return _update_dataframe(values, at, suffix, lkp, amb, kind, col_label)

    needs_alias = any(spec.kind in ("label", "id") for spec in specs)
    if on == "label":
        token_to_id: dict[str, str] | None = build_primary_token_to_id(mapping_set)
        alias_index = (
            build_alias_index(mapping_set, exclude_predicates=relation_predicates)
            if needs_alias
            else {}
        )
    else:
        token_to_id = None
        alias_index = {}
        if needs_alias:
            if alias_mapping_set is None:
                logger.warning(
                    "resolve: a 'label'/'id' context spec was given but no "
                    "'alias_mapping_set' was provided; hint-based resolution is "
                    "limited to direct token/id matches.",
                )
            else:
                alias_index = build_alias_index(
                    alias_mapping_set, exclude_predicates=relation_predicates
                )

    return _update_dataframe_with_context(
        values,
        at,
        suffix,
        lkp,
        amb,
        kind,
        col_label,
        specs=specs,
        alias_index=alias_index,
        token_to_id=token_to_id,
        report_path=report_path,
    )


__all__ = [
    "On",
    "Values",
    "build_alias_index",
    "build_ambiguous",
    "build_lookup",
    "build_primary_token_to_id",
    "resolve",
    "resolve_ambiguous_with_hints",
]
