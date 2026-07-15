"""Tests for the generic SSSOM/RDF/JSON/OWL mapping-set writers."""

from __future__ import annotations

from pathlib import Path

import pytest
from sssom_schema import Mapping

from mapkgsutils.exports import write_json, write_owl, write_rdf, write_sssom
from mapkgsutils.parsers.base import BaseMappingSet

_LICENSE = "https://creativecommons.org/publicdomain/zero/1.0/"


@pytest.fixture
def mapping_set() -> BaseMappingSet:
    """Build a one-row mapping set with cardinalities already computed."""
    ms = BaseMappingSet(
        mapping_set_id="https://example.org/test",
        license=_LICENSE,
        curie_map={
            "CHEBI": "http://purl.obolibrary.org/obo/CHEBI_",
            "IAO": "http://purl.obolibrary.org/obo/IAO_",
            "semapv": "https://w3id.org/semapv/vocab/",
        },
        mappings=[
            Mapping(
                subject_id="CHEBI:1",
                object_id="CHEBI:2",
                predicate_id="IAO:0100001",
                mapping_justification="semapv:BackgroundKnowledgeBasedMatching",
            ),
        ],
    )
    ms._compute_cardinalities(on="id")
    return ms


def test_write_sssom_round_trips_the_mapping(mapping_set: BaseMappingSet, tmp_path: Path) -> None:
    """The written TSV contains the mapping subject/object IDs."""
    out = write_sssom(mapping_set, tmp_path / "out.sssom.tsv")
    text = out.read_text(encoding="utf-8")
    assert "CHEBI:1" in text
    assert "CHEBI:2" in text


def test_write_sssom_keeps_uniform_slots_as_columns(tmp_path: Path) -> None:
    """A slot with the same value on every row is still emitted as a column.

    Write_sssom passes condense=False so provenance columns survive regardless
    of value uniformity.
    """
    ms = BaseMappingSet(
        mapping_set_id="https://example.org/test",
        license=_LICENSE,
        curie_map={
            "CHEBI": "http://purl.obolibrary.org/obo/CHEBI_",
            "IAO": "http://purl.obolibrary.org/obo/IAO_",
            "semapv": "https://w3id.org/semapv/vocab/",
        },
        mappings=[
            Mapping(
                subject_id=f"CHEBI:{i}",
                object_id=f"CHEBI:{i + 1}",
                predicate_id="IAO:0100001",
                mapping_justification="semapv:BackgroundKnowledgeBasedMatching",
                subject_source_version="100",
                object_source_version="100",
            )
            for i in range(5)
        ],
    )
    ms._compute_cardinalities(on="id")

    out = write_sssom(ms, tmp_path / "out.sssom.tsv")
    text = out.read_text(encoding="utf-8")
    data_lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    header_cols = data_lines[0].split("\t")
    assert "subject_source_version" in header_cols
    assert "object_source_version" in header_cols


def test_write_rdf_produces_a_turtle_file(mapping_set: BaseMappingSet, tmp_path: Path) -> None:
    """write_rdf produces a non-empty Turtle file."""
    out = write_rdf(mapping_set, tmp_path / "out.ttl")
    assert out.stat().st_size > 0


def test_write_json_produces_sssom_json(mapping_set: BaseMappingSet, tmp_path: Path) -> None:
    """write_json produces a JSON file mentioning both IDs."""
    out = write_json(mapping_set, tmp_path / "out.json")
    text = out.read_text(encoding="utf-8")
    assert "CHEBI:1" in text
    assert "CHEBI:2" in text


def test_write_owl_produces_a_turtle_file(mapping_set: BaseMappingSet, tmp_path: Path) -> None:
    """write_owl produces a non-empty Turtle file with OWL axioms."""
    out = write_owl(mapping_set, tmp_path / "out_owl.ttl")
    assert out.stat().st_size > 0
