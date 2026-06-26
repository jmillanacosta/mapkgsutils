"""Schema validation for ``config/<datasource>.yaml`` files.

Validates every key used in :class:`~mapkgsutils.parsers.base.DatasourceConfig`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from pathlib import Path


class ConfigValidationError(ValueError):
    """Raised when a datasource config YAML fails schema validation."""

    def __init__(self, file_name: str, message: str) -> None:
        """Store *file_name* so callers can report which config failed."""
        self.file_name = file_name
        super().__init__(f"{file_name}: {message}")


class XrefSourceSchema(BaseModel):
    """Schema for one entry in ``xref_sources``."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = ""
    url: str = ""
    format: str = "tsv"
    object_id_col: str = "object_id"
    object_label_col: str = "object_label"
    subject_id_cols: dict[str, str] = Field(default_factory=dict)
    note: str = ""


class MappingSetEntrySchema(BaseModel):
    """Schema for one entry (e.g. ``ids``/``labels``) in ``mapping_sets``."""

    model_config = ConfigDict(extra="allow")

    method: str | None = None
    primary_input: str | None = None
    required_inputs: list[str] = Field(default_factory=list)
    formats: list[str] = Field(default_factory=list)


class DistributionEraSchema(BaseModel):
    """Schema for one entry in ``distribution_eras``."""

    model_config = ConfigDict(extra="allow")

    id: str = ""
    download_urls: dict[str, str] = Field(default_factory=dict)
    archive_url: str = ""
    format: str | None = None
    from_version: str | None = None
    to_version: str | None = None
    wayback: bool = False


class DatasourceConfigSchema(BaseModel):
    """Top-level schema for a ``config/<datasource>.yaml`` file.

    ``name``, ``prefix``, and ``curie_base_url`` are the only fields
    :class:`~mapkgsutils.parsers.base.DatasourceConfig` declares without a
    default, so they're required here too. Every other recognized key is
    optional and permissive in shape. Unrecognized top-level keys just get a
    warning (see :func:`validate_config_dict`), not a hard error.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    prefix: str
    curie_base_url: str
    config_id: str = ""
    datasource_id: str = ""
    parser_class: str = ""
    entity_types: list[str] = Field(default_factory=list)
    parse_options: dict[str, Any] = Field(default_factory=dict)
    mapping_sets: dict[str, MappingSetEntrySchema] = Field(default_factory=dict)
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
    new_format_version: int | None = None
    distribution_eras: list[DistributionEraSchema] = Field(default_factory=list)
    xref_sources: list[XrefSourceSchema] = Field(default_factory=list)
    species: dict[str, Any] = Field(default_factory=dict)
    subset: dict[str, Any] = Field(default_factory=dict)
    mappingset: dict[str, Any] = Field(default_factory=dict)
    mapping: dict[str, Any] = Field(default_factory=dict)


_KNOWN_TOP_LEVEL_KEYS = set(DatasourceConfigSchema.model_fields)


def validate_config_dict(raw: dict[str, Any], file_name: str) -> DatasourceConfigSchema:
    """Validate a loaded config dict against :class:`DatasourceConfigSchema`.

    Args:
        raw: The dict loaded from a ``config/<datasource>.yaml`` file.
        file_name: Name to attribute errors/warnings to (e.g. ``"hgnc.yaml"``).

    Returns:
        The validated, parsed model.

    Raises:
        ConfigValidationError: When a required field is missing or a known
            field has the wrong shape.
    """
    from mapkgsutils.logging import logger

    unknown = set(raw) - _KNOWN_TOP_LEVEL_KEYS
    if unknown:
        logger.warning(
            "%s: unrecognized top-level key(s) %s (not validated)",
            file_name,
            ", ".join(sorted(unknown)),
        )
    try:
        return DatasourceConfigSchema.model_validate(raw)
    except ValidationError as exc:
        raise ConfigValidationError(file_name, str(exc)) from exc


def validate_config_file(path: Path | str) -> DatasourceConfigSchema:
    """Load and validate a single config YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated, parsed model.

    Raises:
        ConfigValidationError: When the file fails schema validation.
    """
    from pathlib import Path as _Path

    import yaml

    path = _Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return validate_config_dict(raw, path.name)


__all__ = [
    "ConfigValidationError",
    "DatasourceConfigSchema",
    "DistributionEraSchema",
    "MappingSetEntrySchema",
    "XrefSourceSchema",
    "validate_config_dict",
    "validate_config_file",
]
