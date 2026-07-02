"""Generic mapping-set export functions (SSSOM, RDF, JSON, OWL)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from sssom import MappingSetDataFrame

if TYPE_CHECKING:
    from mapkgsutils.parsers.base import BaseMappingSet

_M = TypeVar("_M", bound="BaseMappingSet")

__all__ = [
    "read_sssom",
    "write_json",
    "write_owl",
    "write_rdf",
    "write_sssom",
]


def _mapping_set_from_document(src: Any, converter: Any, mapping_set_class: type[_M]) -> _M:
    """Build *mapping_set_class* from a parsed sssom mapping set and its converter."""
    from dataclasses import fields as dataclass_fields

    kwargs = {
        f.name: v for f in dataclass_fields(src) if (v := getattr(src, f.name, None)) is not None
    }
    kwargs["curie_map"] = dict(converter.prefix_map) or kwargs.get("curie_map") or {}
    return mapping_set_class(**kwargs)


def read_sssom(
    path: Path | str,
    *,
    mapping_set_class: type[_M],
    on: str | None = None,
) -> _M:
    """Load an SSSOM TSV into *mapping_set_class*, a thin wrapper over sssom's parser.

    Carries over the file's mappings, metadata and curie_map. When *on* is
    ``"id"``/``"label"`` cardinalities are recomputed so the result matches a
    freshly parsed set.

    Args:
        path: Path to the SSSOM TSV file.
        mapping_set_class: Concrete :class:`BaseMappingSet` subclass to build.
        on: ``"id"`` or ``"label"`` to recompute cardinalities, or ``None``.

    Returns:
        A ``mapping_set_class`` instance populated from the file.
    """
    from sssom.parsers import parse_sssom_table, to_mapping_set_document

    msdf = parse_sssom_table(str(path))
    src = to_mapping_set_document(msdf).mapping_set
    mapping_set = _mapping_set_from_document(src, msdf.converter, mapping_set_class)
    if on is not None:
        mapping_set._compute_cardinalities(on=on)
    return mapping_set


def write_sssom(
    mapping_set: BaseMappingSet,
    output_path: Path | str,
) -> Path:
    """Write a mapping set to an SSSOM TSV file.

    Args:
        mapping_set: The mapping set to write.
        output_path: Destination ``.sssom.tsv`` file path.

    Returns:
        Path to the written file.
    """
    import codecs
    import re
    from typing import cast

    import curies
    from sssom.parsers import to_mapping_set_dataframe  # type: ignore[attr-defined]
    from sssom.sssom_document import MappingSetDocument
    from sssom.writers import write_table

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a curies.Converter from the curie_map stored on the mapping set.
    # Using Converter(records=...) preserves prefix casing exactly as declared
    # and handles pydantic-wrapped Prefix objects (which expose .prefix_url).
    raw_curie_map: object = mapping_set.curie_map or {}
    records: list[curies.Record] = []
    if isinstance(raw_curie_map, dict):
        for k, v in raw_curie_map.items():
            if isinstance(v, str):
                uri_prefix: str = v
            elif hasattr(v, "prefix_url"):
                uri_prefix = cast(str, v.prefix_url)
            else:
                continue
            records.append(curies.Record(prefix=k, uri_prefix=uri_prefix))
    converter = curies.Converter(records=records)
    doc = MappingSetDocument(mapping_set=mapping_set, converter=converter)
    msdf = to_mapping_set_dataframe(doc)

    # condense=False keeps every per-row slot as an explicit column. With the
    # default condense=True, sssom lifts any slot whose value is identical
    # across all rows into the YAML metadata header, so a consolidated set
    # whose mappings all share one first-seen version/date would silently drop
    # mapping_date/subject_source_version/object_source_version as columns.
    with output_path.open("w", encoding="utf-8") as f:
        write_table(msdf, f, condense=False)

    # Fix escaped unicode in YAML header (sssom issue)
    content = output_path.read_text(encoding="utf-8")
    content = re.sub(
        r"\\x([0-9a-fA-F]{2})",
        lambda m: codecs.decode(bytes([int(m.group(1), 16)]), "latin-1"),
        content,
    )
    output_path.write_text(content, encoding="utf-8")

    return output_path


def _to_msdf_via_sssom_parser(mapping_set: BaseMappingSet) -> MappingSetDataFrame | None:
    """Write to a temporary SSSOM TSV then parse back with sssom's own parser.

    Args:
        mapping_set: The mapping set to convert.

    Returns:
        A fully-validated ``MappingSetDataFrame`` ready for RDF/JSON/OWL serialisation.
    """
    import tempfile

    from sssom.parsers import parse_sssom_table

    with tempfile.NamedTemporaryFile(
        suffix=".sssom.tsv", mode="w", encoding="utf-8", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        write_sssom(mapping_set, tmp_path)
        return parse_sssom_table(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)


def write_rdf(
    mapping_set: BaseMappingSet,
    output_path: Path | str,
    serialisation: str = "turtle",
) -> Path:
    """Write a mapping set to an RDF file.

    Args:
        mapping_set: The mapping set to write.
        output_path: Destination file path (e.g. ``mappings.ttl``).
        serialisation: RDFLib serialisation format.

    Returns:
        Path to the written file.
    """
    from sssom.writers import write_rdf as _sssom_write_rdf

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    msdf = _to_msdf_via_sssom_parser(mapping_set)
    if msdf is None:
        raise ValueError("Failed to parse mapping set for RDF serialisation.")
    with output_path.open("w", encoding="utf-8") as f:
        _sssom_write_rdf(msdf, f, serialisation=serialisation)
    return output_path


def write_json(
    mapping_set: BaseMappingSet,
    output_path: Path | str,
) -> Path:
    """Write a mapping set to an SSSOM JSON file.

    Args:
        mapping_set: The mapping set to write.
        output_path: Destination file path (e.g. ``mappings.json``).

    Returns:
        Path to the written file.
    """
    from sssom.writers import write_json as _sssom_write_json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    msdf = _to_msdf_via_sssom_parser(mapping_set)
    if msdf is None:
        raise ValueError("Failed to parse mapping set for JSON serialisation.")
    with output_path.open("w", encoding="utf-8") as f:
        _sssom_write_json(msdf, f)
    return output_path


def write_owl(
    mapping_set: BaseMappingSet,
    output_path: Path | str,
    serialisation: str = "turtle",
) -> Path:
    """Write a mapping set to an OWL/RDF file (default: Turtle).

    Args:
        mapping_set: The mapping set to write.
        output_path: Destination file path (e.g. ``mappings_owl.ttl``).
        serialisation: RDFLib serialisation format.

    Returns:
        Path to the written file.
    """
    from sssom.writers import write_owl as _sssom_write_owl

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    msdf = _to_msdf_via_sssom_parser(mapping_set)
    if msdf is None:
        raise ValueError("Failed to parse mapping set for OWL serialisation.")
    with output_path.open("w", encoding="utf-8") as f:
        _sssom_write_owl(msdf, f, serialisation=serialisation)
    return output_path
