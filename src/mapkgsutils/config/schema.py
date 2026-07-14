"""Schema validation for ``config/<datasource>.yaml`` files.

Validation and the runtime object share one model:
:class:`~mapkgsutils.parsers.config.DatasourceConfig`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from mapkgsutils.parsers.config import DatasourceConfig

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "ConfigValidationError",
    "validate_config_dict",
    "validate_config_file",
]


class ConfigValidationError(ValueError):
    """Raised when a datasource config YAML fails schema validation."""

    def __init__(self, file_name: str, message: str) -> None:
        """Store *file_name* so callers can report which config failed."""
        self.file_name = file_name
        super().__init__(f"{file_name}: {message}")


def _known_keys() -> set[str]:
    """Field names and aliases recognized by :class:`DatasourceConfig`."""
    keys: set[str] = set()
    for name, info in DatasourceConfig.model_fields.items():
        keys.add(name)
        if info.alias:
            keys.add(info.alias)
    return keys


def validate_config_dict(raw: dict[str, Any], file_name: str) -> DatasourceConfig:
    """Validate a loaded config dict and return the parsed model.

    Args:
        raw: The dict loaded from a ``config/<datasource>.yaml`` file.
        file_name: Name to attribute errors/warnings to (e.g. ``"hgnc.yaml"``).

    Returns:
        The validated :class:`DatasourceConfig`.

    Raises:
        ConfigValidationError: When a required field is missing or a known
            field has the wrong shape.
    """
    from mapkgsutils.logging import logger

    unknown = set(raw) - _known_keys()
    if unknown:
        logger.warning(
            "%s: unrecognized top-level key(s) %s (not validated)",
            file_name,
            ", ".join(sorted(unknown)),
        )
    try:
        return DatasourceConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigValidationError(file_name, str(exc)) from exc


def validate_config_file(path: Path | str) -> DatasourceConfig:
    """Load and validate a single config YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated :class:`DatasourceConfig`.

    Raises:
        ConfigValidationError: When the file fails schema validation.
    """
    from pathlib import Path as _Path

    import yaml

    path = _Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return validate_config_dict(raw, path.name)
