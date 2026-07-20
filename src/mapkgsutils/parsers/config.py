"""Datasource configuration model and YAML loaders.

Split out of :mod:`mapkgsutils.parsers.base` so that config loading (needed by
CLIs to build command help, species choices, etc.) does not drag in the
``sssom_schema``/``linkml`` stack that the mapping-set classes require.

The pydantic models here are both the validation schema and the runtime
objects: :func:`get_datasource_config` returns a validated
:class:`DatasourceConfig` directly.
"""

from __future__ import annotations

from functools import cache
from importlib import resources as _importlib_resources
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DistributionEra(BaseModel):
    """One historical "shape" a datasource's distribution has taken.

    Lets a config describe multiple eras (different URL templates, formats,
    or archive locations) instead of a single hardcoded threshold. Eras are
    matched by version using from_version/to_version (inclusive, numeric-aware
    comparison so "100" < "245" compares correctly; falls back to lexicographic
    for date-string versions like HGNC's "YYYY-MM-DD").
    """

    model_config = ConfigDict(extra="allow")

    id: str = ""
    download_urls: dict[str, str] = Field(default_factory=dict)
    archive_url: str = ""
    format: str | None = None
    from_version: str | None = None
    to_version: str | None = None
    wayback: bool = False


class XrefSource(BaseModel):
    """A suggested cross-reference crosswalk source for a datasource.

    Passed to :func:`mapkgsutils.context.load_xref_mapping` after downloading
    *url* and renaming *object_id_col*/*object_label_col*/the chosen
    *subject_id_cols* entry to ``object_id``/``object_label``/``subject_id``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = ""
    url: str = ""
    format: str = "tsv"
    object_id_col: str = "object_id"
    object_label_col: str = "object_label"
    subject_id_cols: dict[str, str] = Field(default_factory=dict)
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


class DatasourceConfig(BaseModel):
    """Configuration for a biological database datasource loaded from YAML.

    Required fields (``name``, ``prefix``, ``curie_base_url``) have no default;
    every other key is optional. Unrecognized top-level keys are tolerated (a
    warning is emitted by :func:`~mapkgsutils.config.schema.validate_config_dict`).
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str
    prefix: str
    curie_base_url: str
    config_id: str = ""
    datasource_id: str = ""
    parser_class: str = ""
    entity_types: list[str] = Field(default_factory=list)
    parse_options: dict[str, Any] = Field(default_factory=dict)
    mapping_sets: dict[str, Any] = Field(default_factory=dict)
    available_outputs: list[str] = Field(default_factory=list)
    default_output_filename: str = ""
    download_urls: dict[str, Any] = Field(default_factory=dict)
    primary_file_key: str = ""
    id_pattern: str = ""
    archive_url: str | None = ""
    input_file_types: list[str] = Field(default_factory=list)
    source: str = ""
    homepage: str = ""
    data_license: str = ""
    sparql_endpoint: str = ""
    queries: dict[str, str] = Field(default_factory=dict)
    #: SPARQL SELECT run against ``sparql_endpoint`` to get version
    version_query: str = ""
    new_format_version: int | None = None
    distribution_eras: list[DistributionEra] = Field(default_factory=list)
    xref_sources: list[XrefSource] = Field(default_factory=list)
    #: Products split a release into disjoint datasets, e.g.
    #: ``["species"]``. Each names an attribute of the parser; their
    #: values become the product slug
    products: list[str] = Field(default_factory=list)
    species: dict[str, Any] = Field(default_factory=dict)
    subset: dict[str, Any] = Field(default_factory=dict)
    mappingset_metadata: dict[str, Any] = Field(default_factory=dict, alias="mappingset")
    mapping_metadata: dict[str, Any] = Field(default_factory=dict, alias="mapping")

    def species_token(self, taxon_id: str | int) -> str:
        """Resolve a canonical NCBI taxon ID to this datasource's species token.

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


def product_dimensions(config: DatasourceConfig) -> list[str]:
    """Return the names *config* lists under ``products``.

    Each name is an option whose value splits a release into a dataset of its
    own. A source that declares none returns an empty list.
    """
    return list(config.products)


def product_slug_values(config: DatasourceConfig, **options: Any) -> tuple[str, ...]:
    """Return *options*' values for *config*'s dimensions, in declared order.

    These become the slug segments in ``mapping_set_id`` and ``record_id``, and
    the directory names under a cache dir. A dimension the caller left out
    falls back to the config's ``default_<name>()``, then drops out if there is
    no such default.
    """
    values: list[str] = []
    for name in product_dimensions(config):
        value = options.get(name)
        if value is None:
            default = getattr(config, f"default_{name}", None)
            value = default() if callable(default) else None
        if value is not None:
            values.append(str(value))
    return tuple(values)


@cache
def load_config(datasource_name: str, *, config_package: str) -> dict[str, Any]:
    """Load the raw YAML config dict for a datasource.

    The result is cached per ``(datasource_name, config_package)``: a single
    CLI invocation otherwise re-reads the same YAML dozens of times. Callers
    must treat the returned dict as read-only.

    Args:
        datasource_name: Name of the datasource (e.g., 'chebi', 'hgnc').
        config_package: Importable package holding the datasource's
            ``*.yaml`` config files (e.g. ``"pysec2pri.config"``).

    Returns:
        Dictionary with the full YAML configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_dir = Path(_importlib_resources.files(config_package))  # type: ignore[arg-type]
    config_path = config_dir / f"{datasource_name.lower()}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        result: dict[str, Any] = yaml.safe_load(f)
    return result


def get_datasource_config(datasource_name: str, *, config_package: str) -> DatasourceConfig:
    """Load, validate, and return a :class:`DatasourceConfig` from YAML.

    Args:
        datasource_name: Name of the datasource (e.g., 'chebi', 'hgnc').
        config_package: Importable package holding the datasource's
            ``*.yaml`` config files (e.g. ``"pysec2pri.config"``).

    Returns:
        Validated DatasourceConfig object populated from YAML.
    """
    from mapkgsutils.config.schema import validate_config_dict

    raw = load_config(datasource_name, config_package=config_package)
    return validate_config_dict(raw, f"{datasource_name.lower()}.yaml")
