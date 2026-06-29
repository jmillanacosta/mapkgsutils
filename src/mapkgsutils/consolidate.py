"""Generic mapping-date consolidation: cache I/O and the release/date walk skeleton.

A versioned datasource doesn't always record when a mapping first appeared.
Recovering that date means walking every historical release once and
recording the first/last release each mapping (keyed by its
version-independent pair hash) was seen in.

This module provides the cache I/O, the release-walk and single-pass-date
loop shapes, and rebuilding the cache back into a real SSSOM mapping set.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from mapkgsutils.logging import logger
from mapkgsutils.parsers.base import _cmp_versions

__all__ = [
    "CACHE_COLUMNS",
    "build_consolidated_mapping_set",
    "consolidate_by_date",
    "consolidate_by_release",
    "load_mapping_dates",
    "read_cache",
    "read_meta",
    "sssom_output_path",
    "write_cache",
    "write_consolidated_sssom",
    "write_meta",
]

CACHE_COLUMNS = (
    "record_id",
    "first_seen_version",
    "first_seen_date",
    "last_seen_version",
    "last_seen_date",
    # JSON-encoded snapshot of the mapping's own fields (subject_id,
    # object_id, predicate_id, ...) as last seen. Used to rebuild the cache
    # into a real SSSOM mapping set later (see build_consolidated_mapping_set).
    # Empty for legacy/hand-built rows.
    "fields_json",
)


def sssom_output_path(cache_path: Path) -> Path:
    """Return the companion SSSOM mapping-set path for a consolidated cache file."""
    return cache_path.with_name(cache_path.stem + "_sssom.tsv")


def read_cache(cache_path: Path) -> dict[str, dict[str, str]]:
    """Read a consolidated mapping-date cache TSV into a dict keyed by record_id."""
    if not cache_path.exists():
        return {}

    import polars as pl

    df = pl.read_csv(cache_path, separator="\t", schema_overrides={"record_id": pl.Utf8})
    cols = [c for c in CACHE_COLUMNS[1:] if c in df.columns]
    return {
        str(row["record_id"]): {col: str(row[col]) for col in cols}
        for row in df.iter_rows(named=True)
    }


def write_cache(cache_path: Path, records: dict[str, dict[str, str]]) -> None:
    """Write the merged ``record_id -> first/last seen`` dict to a TSV file."""
    import polars as pl

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"record_id": rid, **{c: fields.get(c, "") for c in CACHE_COLUMNS[1:]}}
        for rid, fields in records.items()
    ]
    schema = list(CACHE_COLUMNS)
    df = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
    df.write_csv(cache_path, separator="\t")


def read_meta(meta_path: Path) -> str | None:
    """Read the ``last_version`` sidecar, or ``None`` if absent/unreadable."""
    if not meta_path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    last_version = data.get("last_version")
    return str(last_version) if last_version is not None else None


def write_meta(meta_path: Path, last_version: str) -> None:
    """Write the ``last_version`` sidecar used to resume an interrupted walk."""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"last_version": last_version}), encoding="utf-8")


def load_mapping_dates(cache_path: Path) -> dict[str, str]:
    """Load the consolidated ``record_id -> first_seen_date`` index at *cache_path*.

    Safe to call even when the index hasn't been built yet: returns ``{}``.

    Args:
        cache_path: Path to the cache TSV (see :func:`write_cache`).

    Returns:
        Dict mapping each ``record_id`` to its first-seen ISO date string.
        Records walked but never assigned a real release date (e.g. an
        unresolvable ``Last-Modified``) are omitted, leaving their
        ``mapping_date`` unset rather than passing through a non-date value.
    """
    records = read_cache(cache_path)
    return {
        rid: fields["first_seen_date"]
        for rid, fields in records.items()
        if fields.get("first_seen_date")
    }


def _mapping_fields_json(m: Any) -> str:
    """JSON-encode a mapping's own fields (excluding mapping_date/record_id).

    Snapshots a mapping's current shape (subject_id, object_id,
    predicate_id, ...) so :func:`build_consolidated_mapping_set` can rebuild
    it into a real SSSOM mapping set later. ``default=str`` handles
    enum-valued fields like ``mapping_cardinality``, which round-trip
    cleanly back through ``Mapping(**fields)``. Returns ``"{}"`` for
    non-dataclass stand-ins (e.g. test doubles); callers skip those rows
    rather than materializing them.
    """
    from dataclasses import fields as dataclass_fields
    from dataclasses import is_dataclass

    if not is_dataclass(m):
        return "{}"
    fields = {
        f.name: getattr(m, f.name)
        for f in dataclass_fields(m)
        if f.name not in ("mapping_date", "record_id") and getattr(m, f.name, None) is not None
    }
    return json.dumps(fields, default=str)


def consolidate_by_date(
    cache_path: Path,
    meta_path: Path,
    *,
    run_one_version: Callable[[], Any],
) -> bool:
    """Single-pass "date" mode: capture each row's own real ``mapping_date``.

    Args:
        cache_path: Path to the cache TSV to write.
        meta_path: Path to the ``last_version`` sidecar to write.
        run_one_version: Zero-arg callable returning a freshly parsed
            mapping set (the current/latest release, already bound to
            whatever datasource/mapping-kind/kwargs the caller needs).

    Returns:
        ``True`` and writes the cache when the mapping set produced at
        least one real per-row date; ``False`` (cache untouched) when it
        produced none, so the caller can fall back to a release-walk.
    """
    mapping_set = run_one_version()
    dated = [m for m in (mapping_set.mappings or []) if getattr(m, "mapping_date", None)]
    if not dated:
        return False

    version_label = str(getattr(mapping_set, "mapping_set_version", None) or "current")
    records: dict[str, dict[str, str]] = {}
    for m in dated:
        # record_id is release-scoped; its trailing 16 hex chars are always
        # the version-independent pair hash (see mapkgsutils.parsers.base.pair_hash),
        # which is what must match across releases.
        pair_key = str(getattr(m, "record_id", None) or "")[-16:]
        if not pair_key:
            continue
        date_str = str(m.mapping_date)
        records[pair_key] = {
            "first_seen_version": version_label,
            "first_seen_date": date_str,
            "last_seen_version": version_label,
            "last_seen_date": date_str,
            "fields_json": _mapping_fields_json(m),
        }
    write_cache(cache_path, records)
    write_meta(meta_path, version_label)
    return True


def build_consolidated_mapping_set(
    records: dict[str, dict[str, str]],
    last_version: str | None,
    *,
    mapping_set_class: type[Any],
    record_namespace: str,
    mapping_set_metadata: Mapping[str, Any],
    cardinality_on: str = "id",
) -> Any:
    """Materialize the consolidated index as a real SSSOM mapping set.

    Each row's ``mapping_date`` and ``subject_source_version``/
    ``object_source_version`` are overridden from the record's
    ``first_seen_date``/``first_seen_version``: the date of first
    appearance and the release it first appeared in, rather than whatever
    the last-seen snapshot happened to carry. The two stay in their own
    fields. A release version (e.g. ChEBI's "183") is not a date and never
    goes in ``mapping_date``, and a real date never goes in the version
    fields.

    Args:
        records: ``record_id -> fields`` dict as read by :func:`read_cache`.
        last_version: The most recent release the walk has processed.
        mapping_set_class: Concrete ``BaseMappingSet`` subclass to build.
        record_namespace: Base IRI namespace prefixed to each rebuilt
            ``record_id``.
        mapping_set_metadata: ``mapping_set_id``/``curie_map``/``license``/...
            metadata for the resulting mapping set.
        cardinality_on: ``"id"`` or ``"label"``, forwarded to
            :meth:`~mapkgsutils.parsers.base.BaseMappingSet._compute_cardinalities`.

    Returns:
        A ``mapping_set_class`` instance with cardinalities computed.
    """
    from sssom_schema import Mapping as SSSOMMapping

    version_label = str(last_version) if last_version else "current"

    mappings = []
    for pair_key, fields in records.items():
        fields_json = fields.get("fields_json") or ""
        if not fields_json:
            continue
        try:
            row_fields = json.loads(fields_json)
        except json.JSONDecodeError:
            continue
        if not row_fields:
            # Non-dataclass stand-ins (see _mapping_fields_json) have
            # nothing to materialize into a real Mapping; date bookkeeping
            # for them still lives in the TSV cache, just not in the SSSOM.
            continue
        # mapping_date and subject/object_source_version are semantically
        # distinct SSSOM fields: a release version (e.g. ChEBI's "183") is
        # never a date, and must never end up in mapping_date or vice versa.
        # Override both from first_seen_*, not whatever the snapshot's own
        # (last-seen) values were.
        row_fields["mapping_date"] = fields.get("first_seen_date") or None
        first_seen_version = fields.get("first_seen_version") or None
        row_fields["subject_source_version"] = first_seen_version
        row_fields["object_source_version"] = first_seen_version
        row_fields["record_id"] = f"{record_namespace}{version_label}/consolidate/{pair_key}"
        mappings.append(SSSOMMapping(**row_fields))

    base_ms_id = str(mapping_set_metadata.get("mapping_set_id") or "")
    mapping_set = mapping_set_class(
        mappings=mappings,
        curie_map=mapping_set_metadata.get("curie_map") or {},
        mapping_set_id=f"{base_ms_id}/{version_label}/consolidate",
        mapping_set_version=version_label,
        mapping_set_title=mapping_set_metadata.get("mapping_set_title"),
        mapping_set_description=mapping_set_metadata.get("mapping_set_description"),
        license=mapping_set_metadata.get("license"),
    )
    mapping_set._compute_cardinalities(on=cardinality_on)
    return mapping_set


def write_consolidated_sssom(
    cache_path: Path,
    meta_path: Path,
    *,
    mapping_set_class: type[Any],
    record_namespace: str,
    mapping_set_metadata: Mapping[str, Any],
    cardinality_on: str = "id",
) -> tuple[Path, Any]:
    """Build and save the companion SSSOM mapping set next to the cache file.

    Args:
        cache_path: Path to the cache TSV (see :func:`read_cache`).
        meta_path: Path to the ``last_version`` sidecar (see :func:`read_meta`).
        mapping_set_class: Concrete ``BaseMappingSet`` subclass to build.
        record_namespace: Base IRI namespace for rebuilt ``record_id`` values.
        mapping_set_metadata: ``mapping_set_id``/``curie_map``/``license``/...
            metadata for the resulting mapping set.
        cardinality_on: ``"id"`` or ``"label"``, see
            :func:`build_consolidated_mapping_set`.

    Returns:
        ``(output_path, mapping_set)``: the path of the written SSSOM TSV
        (see :func:`sssom_output_path`) and the in-memory mapping set.
    """
    records = read_cache(cache_path)
    last_version = read_meta(meta_path)
    mapping_set = build_consolidated_mapping_set(
        records,
        last_version,
        mapping_set_class=mapping_set_class,
        record_namespace=record_namespace,
        mapping_set_metadata=mapping_set_metadata,
        cardinality_on=cardinality_on,
    )
    output_path = sssom_output_path(cache_path)
    mapping_set.save("sssom", output_path)
    return output_path, mapping_set


def consolidate_by_release(
    cache_path: Path,
    meta_path: Path,
    *,
    label: str,
    list_versions: Callable[[], list[str]],
    run_one_version: Callable[[str], Any],
    resolve_release_date: Callable[[str], datetime | None],
    show_progress: bool = True,
    force: bool = False,
) -> None:
    """Historical-walk "release" mode: track first/last-seen release per mapping.

    Args:
        cache_path: Path to the cache TSV to read/write.
        meta_path: Path to the ``last_version`` sidecar to read/write.
        label: Datasource name, used only for the progress bar/log messages.
        list_versions: Zero-arg callable returning every available version,
            oldest first.
        run_one_version: Callable taking a version string and returning a
            freshly downloaded+parsed mapping set for it.
        resolve_release_date: Callable taking a version string and
            returning its upstream release date, or ``None`` when
            unresolvable.
        show_progress: Whether to show a progress bar over releases.
        force: Re-scan every release from scratch, ignoring any existing
            cache/resume state.
    """
    records: dict[str, dict[str, str]] = {} if force else read_cache(cache_path)
    last_version = None if force else read_meta(meta_path)

    versions = list_versions()
    if last_version is not None:
        versions = [v for v in versions if _cmp_versions(v, last_version) > 0]

    iterator: Iterable[str] = versions
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(versions, desc=f"Consolidating {label.upper()} mapping dates")

    for v in iterator:
        try:
            mapping_set = run_one_version(v)
            release_date = resolve_release_date(v)
            # Empty when no real release date resolves. v is a version, not
            # a date, and isn't always date-shaped (e.g. ChEBI's plain
            # release numbers), so don't store it as one.
            date_str = release_date.date().isoformat() if release_date else ""

            for m in mapping_set.mappings or []:
                # record_id is release-scoped; match across releases on its
                # trailing pair hash instead (see mapkgsutils.parsers.base.pair_hash).
                pair_key = str(getattr(m, "record_id", None) or "")[-16:]
                if not pair_key:
                    continue
                entry = records.setdefault(
                    pair_key,
                    {"first_seen_version": v, "first_seen_date": date_str},
                )
                entry["last_seen_version"] = v
                entry["last_seen_date"] = date_str
                entry["fields_json"] = _mapping_fields_json(m)
        except Exception:
            logger.warning("Skipping %s version %s during consolidation", label, v, exc_info=True)
            continue

        last_version = v
        write_cache(cache_path, records)
        write_meta(meta_path, last_version)
