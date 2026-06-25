"""Tests for datasource config schema validation."""

from __future__ import annotations

import pytest

from mapkgsutils.config.schema import ConfigValidationError, validate_config_dict


def test_broken_config_fails_validation() -> None:
    """A config dict missing a required field is rejected, not silently accepted."""
    with pytest.raises(ConfigValidationError):
        broken = {"prefix": "X", "curie_base_url": "http://example.org/"}
        validate_config_dict(broken, "broken.yaml")
