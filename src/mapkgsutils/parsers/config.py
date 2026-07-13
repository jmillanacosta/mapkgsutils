"""Datasource configuration model and YAML loaders.

Split out of :mod:`mapkgsutils.parsers.base` so that config loading (needed by
CLIs to build command help, species choices, etc.) does not drag in the
``sssom_schema``/``linkml`` stack that the mapping-set classes require.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from importlib import resources as _importlib_resources
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass
class DistributionEra:
    """One historical "shape" a datasource's distribution has taken.

    Lets a config describe multiple eras (different URL templates, formats,
    or archive locations) instead of a single hardcoded threshold. Eras are
    matched by version using from_version/to_version (inclusive, numeric-aware
    comparison so "100" < "245" compares correctly; falls back to lexicographic
    for date-string versions like HGNC's "YYYY-MM-DD").
    """

    id: str
    download_urls: dict[str, str] = field(default_factory=dict)
    archive_url: str = ""
    format: str | None = None
    from_version: str | None = None
    to_version: str | None = None
    wayback: bool = False  # declarative only for now, no resolver yet


@dataclass
class XrefSource:
    """A suggested cross-reference crosswalk source for a datasource.

    Passed to :func:`mapkgsutils.context.load_xref_mapping` after downloading
    *url* and renaming *object_id_col*/*object_label_col*/the chosen
    *subject_id_cols* entry to ``object_id``/``object_label``/``subject_id``.
    """

    id: str
    name: str = ""
    url: str = ""
    format: str = "tsv"
    object_id_col: str = "object_id"
    object_label_col: str = "object_label"
    subject_id_cols: dict[str, str] = field(default_factory=dict)
    note: str = ""


def _cmp_versions(a: str, b: str) -> int:
    """Compare two version strings, numerically when possible.

    Falls back to plain string comparison for non-numeric versions (e.g.
    ISO date strings like ``"2026-04-07"``, which already sort correctly
    lexicographically).

    Returns:
        Negative if ``a < b``, zero if equal, positive if ``a > b``.
    """
    try:
        ai, bi = int(a), int(b)
        return (ai > bi) - (ai < bi)
    except ValueError:
        return (a > b) - (a < b)


@dataclass
class DatasourceConfig:
    """Configuration for a biological database datasource loaded from YAML."""

    name: str
    prefix: str
    curie_base_url: str
    # Proper schema fields
    config_id: str = ""
    datasource_id: str = ""
    parser_class: str = ""
    parse_options: dict[str, Any] = field(default_factory=dict)
    mapping_sets: dict[str, Any] = field(default_factory=dict)
    # Old field, remove at some point
    available_outputs: list[str] = field(default_factory=list)
    default_output_filename: str = ""
    download_urls: dict[str, Any] = field(default_factory=dict)
    primary_file_key: str = ""
    id_pattern: str = ""
    archive_url: str = ""
    input_file_types: list[str] = field(default_factory=list)
    source: str = ""
    homepage: str = ""
    data_license: str = ""
    # SPARQL-based datasources (e.g., Wikidata)
    sparql_endpoint: str = ""
    queries: dict[str, str] = field(default_factory=dict)
    # For now, only ChEBI: version threshold for new TSV format.
    # Use if release files change location or serialization.
    new_format_version: int | None = None
    # Historical distribution "shapes" this datasource has had (different URL
    # templates, formats, or archive locations across its lifetime).
    distribution_eras: list[DistributionEra] = field(default_factory=list)
    # Suggested cross-reference crosswalk sources
    xref_sources: list[XrefSource] = field(default_factory=list)
    # Species this datasource publishes
    species: dict[str, Any] = field(default_factory=dict)
    # Compound/entry subset this datasource publishes (e.g. ChEBI's
    # 3star/complete). Generic, config-driven counterpart to `species`.
    subset: dict[str, Any] = field(default_factory=dict)
    # Full metadata from YAML
    mappingset_metadata: dict[str, Any] = field(default_factory=dict)
    mapping_metadata: dict[str, Any] = field(default_factory=dict)

    def species_token(self, taxon_id: str | int) -> str:
        """Resolve a canonical NCBI taxon ID to this datasource's own species token.

        Reads the ``species.available`` block (see ``ensembl.yaml``), which
        maps each supported taxon ID to the datasource-specific token used to
        build download paths/filters (e.g. Ensembl's ``homo_sapiens``).

        Args:
            taxon_id: Canonical NCBI taxon ID, e.g. ``9606`` or ``"9606"``.

        Returns:
            The datasource-specific species token.

        Raises:
            ValueError: If no ``species`` block is configured, or *taxon_id*
                is not one of its declared entries.
        """
        available = {str(k): v for k, v in ((self.species or {}).get("available") or {}).items()}
        entry = available.get(str(taxon_id))
        if entry is None:
            known = ", ".join(sorted(available)) or "(none configured)"
            raise ValueError(
                f"Unknown species taxon ID {taxon_id!r} for {self.name!r}. Known: {known}"
            )
        return str(entry["token"])

    def default_species(self) -> str | int:
        """Return the configured default species taxon ID (``9606`` if unset)."""
        return cast("str | int", (self.species or {}).get("default", 9606))

    def default_subset(self) -> str | None:
        """Return the configured default subset, or ``None`` if this datasource has none."""
        return cast("str | None", (self.subset or {}).get("default"))

    def xref_source(self, source_id: str) -> XrefSource | None:
        """Return the configured :class:`XrefSource` with id *source_id*, if any."""
        for src in self.xref_sources:
            if src.id == source_id:
                return src
        return None

    def formats_for(self, kind: str) -> Any:
        """Return the list of supported output formats for a mapping-set kind.

        Args:
            kind: Mapping-set key, e.g. ``"ids"`` or ``"labels"``.

        Returns:
            List of format strings, or an empty list when the kind is absent.
        """
        return self.mapping_sets.get(kind, {}).get("formats", [])

    def era_for(self, version: str | None) -> DistributionEra | None:
        """Return the first configured era whose bounds contain *version*.

        Args:
            version: Version string to match, or ``None``.

        Returns:
            The matching :class:`DistributionEra`, or ``None`` if no eras are
            configured or none match (callers should fall back to the
            top-level ``download_urls``/``new_format_version`` behavior).
        """
        if not self.distribution_eras or version is None:
            return None
        for era in self.distribution_eras:
            if era.from_version is not None and _cmp_versions(version, era.from_version) < 0:
                continue
            if era.to_version is not None and _cmp_versions(version, era.to_version) > 0:
                continue
            return era
        return None


@cache
def load_config(datasource_name: str, *, config_package: str) -> dict[str, Any]:
    """Load configuration from a YAML file for a datasource.

    The result is cached per ``(datasource_name, config_package)``: a single
    CLI invocation otherwise re-reads and re-validates the same YAML dozens of
    times. Callers must treat the returned dict as read-only.

    Args:
        datasource_name: Name of the datasource (e.g., 'chebi', 'hgnc').
        config_package: Importable package holding the datasource's
            ``*.yaml`` config files (e.g. ``"pysec2pri.config"``).

    Returns:
        Dictionary with the full YAML configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    from mapkgsutils.config.schema import validate_config_dict

    config_dir = Path(_importlib_resources.files(config_package))  # type: ignore[arg-type]
    config_path = config_dir / f"{datasource_name.lower()}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        result: dict[str, Any] = yaml.safe_load(f)
    validate_config_dict(result, config_path.name)
    return result


def get_datasource_config(datasource_name: str, *, config_package: str) -> DatasourceConfig:
    """Load and parse a DatasourceConfig from YAML.

    Args:
        datasource_name: Name of the datasource (e.g., 'chebi', 'hgnc').
        config_package: Importable package holding the datasource's
            ``*.yaml`` config files (e.g. ``"pysec2pri.config"``).

    Returns:
        DatasourceConfig object populated from YAML.
    """
    raw = load_config(datasource_name, config_package=config_package)

    eras = [
        DistributionEra(
            id=era.get("id", ""),
            download_urls=era.get("download_urls") or {},
            archive_url=era.get("archive_url", ""),
            format=era.get("format"),
            from_version=era.get("from_version"),
            to_version=era.get("to_version"),
            wayback=era.get("wayback", False),
        )
        for era in raw.get("distribution_eras", [])
    ]

    xref_sources = [
        XrefSource(
            id=src.get("id", ""),
            name=src.get("name", ""),
            url=src.get("url", ""),
            format=src.get("format", "tsv"),
            object_id_col=src.get("object_id_col", "object_id"),
            object_label_col=src.get("object_label_col", "object_label"),
            subject_id_cols=src.get("subject_id_cols") or {},
            note=src.get("note", ""),
        )
        for src in raw.get("xref_sources", [])
    ]

    return DatasourceConfig(
        name=raw.get("name", ""),
        prefix=raw.get("prefix", ""),
        curie_base_url=raw.get("curie_base_url", ""),
        config_id=raw.get("config_id", ""),
        datasource_id=raw.get("datasource_id", ""),
        parser_class=raw.get("parser_class", ""),
        parse_options=raw.get("parse_options") or {},
        mapping_sets=raw.get("mapping_sets") or {},
        available_outputs=raw.get("available_outputs", []),
        default_output_filename=raw.get("default_output_filename", ""),
        download_urls=raw.get("download_urls", {}),
        primary_file_key=raw.get("primary_file_key", ""),
        id_pattern=raw.get("id_pattern", ""),
        archive_url=raw.get("archive_url", ""),
        input_file_types=raw.get("input_file_types", []),
        source=raw.get("source", ""),
        homepage=raw.get("homepage", ""),
        data_license=raw.get("data_license", ""),
        sparql_endpoint=raw.get("sparql_endpoint", ""),
        queries=raw.get("queries", {}),
        new_format_version=raw.get("new_format_version"),
        distribution_eras=eras,
        xref_sources=xref_sources,
        species=raw.get("species") or {},
        subset=raw.get("subset") or {},
        mappingset_metadata=raw.get("mappingset", {}),
        mapping_metadata=raw.get("mapping", {}),
    )
