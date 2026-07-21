"""Tests for the generic parser/downloader framework and mapping-set classes."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from sssom_schema import Mapping

from mapkgsutils.parsers.base import (
    BaseMappingSet,
    BaseParser,
    _cmp_versions,
    mint_record_id,
    pair_hash,
)
from mapkgsutils.parsers.config import DatasourceConfig

_LICENSE = "https://creativecommons.org/publicdomain/zero/1.0/"


def _mapping(subject_id: str, object_id: str, **kwargs: object) -> Mapping:
    return Mapping(
        subject_id=subject_id,
        object_id=object_id,
        predicate_id="IAO:0100001",
        mapping_justification="semapv:BackgroundKnowledgeBasedMatching",
        **kwargs,
    )


class TestCmpVersions:
    """Version comparison."""

    def test_non_numeric_versions_fall_back_to_string_compare(self) -> None:
        """ISO dates sort correctly as plain strings."""
        assert _cmp_versions("2024-01-01", "2024-04-01") < 0
        assert _cmp_versions("2024_02", "2024_01") > 0


class TestPairHashAndRecordId:
    """pair_hash is the version-independent join key; mint_record_id scopes it."""

    def test_pair_hash_is_stable_for_the_same_pair(self) -> None:
        """The same (pri, sec) pair always hashes the same."""
        assert pair_hash("CHEBI:1", "CHEBI:2") == pair_hash("CHEBI:1", "CHEBI:2")

    def test_pair_hash_differs_for_different_pairs(self) -> None:
        """Different pairs hash differently."""
        assert pair_hash("CHEBI:1", "CHEBI:2") != pair_hash("CHEBI:1", "CHEBI:3")

    def test_mint_record_id_ends_with_the_pair_hash(self) -> None:
        """record_id is namespace + pair_hash, so its trailing 16 chars never change."""
        rid = mint_record_id("CHEBI:1", "CHEBI:2", namespace="sec2pri:chebi/245/")
        assert rid == "sec2pri:chebi/245/" + pair_hash("CHEBI:1", "CHEBI:2")

    def test_mint_record_id_differs_across_namespaces_but_pair_hash_does_not(self) -> None:
        """Two releases of the same pair get different record_ids, same trailing hash."""
        older = mint_record_id("CHEBI:1", "CHEBI:2", namespace="sec2pri:chebi/200/")
        newer = mint_record_id("CHEBI:1", "CHEBI:2", namespace="sec2pri:chebi/245/")
        assert older != newer
        assert older[-16:] == newer[-16:] == pair_hash("CHEBI:1", "CHEBI:2")


class TestBaseMappingSet:
    """BaseMappingSet: cardinalities, save dispatch, and SSSOM export."""

    def test_compute_cardinalities_by_id(self) -> None:
        """A primary ID shared by two secondaries is flagged n:1."""
        ms = BaseMappingSet(
            mapping_set_id="https://example.org/test",
            license="https://creativecommons.org/publicdomain/zero/1.0/",
            mappings=[
                _mapping("SEC:1", "PRI:1"),
                _mapping("SEC:2", "PRI:1"),
            ],
        )
        ms._compute_cardinalities(on="id")
        assert [str(m.mapping_cardinality) for m in ms.mappings] == ["n:1", "n:1"]

    def test_save_sssom_writes_a_tsv_file(self, tmp_path: Path) -> None:
        """save("sssom", ...) writes a real SSSOM TSV to disk."""
        ms = BaseMappingSet(
            mapping_set_id="https://example.org/test",
            license="https://creativecommons.org/publicdomain/zero/1.0/",
            mappings=[_mapping("SEC:1", "PRI:1")],
        )
        ms._compute_cardinalities(on="id")
        out = ms.save("sssom", tmp_path / "out.sssom.tsv")
        assert out.exists()
        assert "SEC:1" in out.read_text(encoding="utf-8")

    def test_save_unknown_format_raises(self, tmp_path: Path) -> None:
        """An unsupported format name raises."""
        ms = BaseMappingSet(
            mapping_set_id="https://example.org/test", license=_LICENSE, mappings=[]
        )
        try:
            ms.save("not-a-real-format", tmp_path / "out")
        except ValueError as exc:
            assert "Unknown format" in str(exc)
        else:
            raise AssertionError("expected ValueError")


class TestFindAmbiguous:
    """The ambiguity engine: a secondary that's also a live primary is flagged."""

    def test_id_mapping_flagged_when_subject_is_also_a_primary(self) -> None:
        """SEC:1 -> PRI:1, but SEC:1 is itself a primary elsewhere: ambiguous."""
        ms = BaseMappingSet(
            mapping_set_id="https://example.org/test",
            license=_LICENSE,
            mappings=[
                _mapping("SEC:1", "PRI:1"),
                _mapping("SEC:2", "SEC:1"),
            ],
        )
        ambiguous = ms.find_ambiguous()
        flagged = {m.subject_id for m in ambiguous.mappings}
        assert flagged == {"SEC:1"}
        assert "SEC:1" in ambiguous.ambiguous_ids

    def test_no_ambiguity_when_secondaries_and_primaries_are_disjoint(self) -> None:
        """No secondary doubles as a primary: nothing comes back ambiguous."""
        ms = BaseMappingSet(
            mapping_set_id="https://example.org/test",
            license=_LICENSE,
            mappings=[
                _mapping("SEC:1", "PRI:1"),
                _mapping("SEC:2", "PRI:2"),
            ],
        )
        ambiguous = ms.find_ambiguous()
        assert ambiguous.mappings == []

    def test_label_mapping_flagged_when_subject_label_is_also_a_primary_label(self) -> None:
        """A previous label that's also someone else's current label is ambiguous."""

        class _LabelMappingSet(BaseMappingSet):
            _ambiguity_mode: ClassVar[str] = "label"

        ms = _LabelMappingSet(
            mapping_set_id="https://example.org/test",
            license=_LICENSE,
            mappings=[
                _mapping("SEC:1", "PRI:1", subject_label="OLD", object_label="NEW"),
                _mapping("SEC:2", "PRI:2", subject_label="X", object_label="OLD"),
            ],
        )
        ambiguous = ms.find_ambiguous()
        flagged = {m.subject_label for m in ambiguous.mappings}
        assert flagged == {"OLD"}
        assert "OLD" in ambiguous.ambiguous_labels


class _ToyParser(BaseParser):
    """A minimal concrete BaseParser, with no datasource config, for framework tests."""

    def parse(self, input_path: Path | str | None) -> BaseMappingSet:
        """Not exercised by these tests; required to make the class concrete."""
        raise NotImplementedError


class TestBaseParserFramework:
    """BaseParser helpers."""

    def test_pair_hash_and_record_id_delegate_to_the_module_functions(self) -> None:
        """BaseParser._pair_hash/_record_id are thin wrappers, not a separate hash."""
        parser = _ToyParser(version="1")
        assert parser._pair_hash("A", "B") == pair_hash("A", "B")
        assert parser._record_id("ns/", "A", "B") == mint_record_id("A", "B", namespace="ns/")

    def test_record_namespace_folds_in_version_and_product_slug(self) -> None:
        """_record_namespace appends version/slug, mirroring mapping_set_id's layout."""
        parser = _ToyParser(version="245")
        assert parser._record_namespace() == "245/"

        class _SlugParser(_ToyParser):
            def _product_slug(self) -> str | None:
                return "9606"

        slugged = _SlugParser(version="245")
        assert slugged._record_namespace() == "245/9606/"

    def test_product_slug_override_beats_the_instance_attribute(self) -> None:
        """A per-call override wins over self.<dimension>, for per-row product resolution.

        Needed by datasources like NCBI, where a single "species=all" parse
        spans every organism at once: self.species is the collection
        selector "all", not any one row's actual value, so each row must
        resolve its own slug via an override instead of the instance-wide
        attribute.
        """
        parser = _ToyParser(version="245")
        parser._config = DatasourceConfig(
            name="Test", prefix="TEST", curie_base_url="http://example.org/", products=["species"]
        )
        parser.species = "all"

        assert parser._product_slug() == "all"
        assert parser._product_slug(species="9606") == "9606"
        assert parser._record_namespace(species="9606") == "245/9606/"

    def test_create_mapping_set_computes_cardinalities_and_id_for_id_type(self) -> None:
        """create_mapping_set builds a mapping set with cardinalities already set."""

        class _ConfiguredParser(_ToyParser):
            def get_mappingset_metadata(self) -> dict[str, object]:
                """Return the minimal metadata create_mapping_set needs."""
                return {"mapping_set_id": "https://example.org/test", "license": _LICENSE}

        parser = _ConfiguredParser(version="1")
        mappings = [_mapping("SEC:1", "PRI:1"), _mapping("SEC:2", "PRI:1")]
        ms = parser.create_mapping_set(mappings, mapping_type="id")
        assert isinstance(ms, BaseMappingSet)
        assert [str(m.mapping_cardinality) for m in ms.mappings] == ["n:1", "n:1"]

    def test_mapping_set_title_gets_product_label_and_version(self) -> None:
        """Title names the product (via its configured label) and version, not a static string."""
        parser = _ToyParser(version="2026-07-21")
        parser._config = DatasourceConfig(
            name="NCBI Gene",
            prefix="TEST",
            curie_base_url="http://example.org/",
            products=["species"],
            species={"available": {"9606": {"label": "Human"}}},
            mappingset={
                "mapping_set_id": "https://example.org/test",
                "mapping_set_title": "NCBI Gene Secondary to Primary Mapping",
                "license": _LICENSE,
            },
        )
        parser.species = "9606"

        ms = parser.create_mapping_set([], mapping_type="id")

        assert ms.mapping_set_title == (
            "NCBI Gene Secondary to Primary Mapping (Human, 2026-07-21)"
        )

    def test_mapping_set_title_falls_back_to_raw_value_when_unlabeled(self) -> None:
        """A product value outside the curated shortlist still appears, just unlabeled."""
        parser = _ToyParser(version="2026-07-21")
        parser._config = DatasourceConfig(
            name="NCBI Gene",
            prefix="TEST",
            curie_base_url="http://example.org/",
            products=["species"],
            species={"available": {"9606": {"label": "Human"}}},
            mappingset={
                "mapping_set_id": "https://example.org/test",
                "mapping_set_title": "NCBI Gene Secondary to Primary Mapping",
                "license": _LICENSE,
            },
        )
        parser.species = "10090"  # not in the curated shortlist

        ms = parser.create_mapping_set([], mapping_type="id")

        assert ms.mapping_set_title == "NCBI Gene Secondary to Primary Mapping (10090, 2026-07-21)"

    def test_mapping_set_title_with_no_products_just_gets_version(self) -> None:
        """A datasource with no product dimensions (e.g. HGNC) still gets the version suffix."""

        class _ConfiguredParser(_ToyParser):
            def get_mappingset_metadata(self) -> dict[str, object]:
                """Return minimal metadata with no product dimensions declared."""
                return {
                    "mapping_set_id": "https://example.org/test",
                    "mapping_set_title": "HGNC Mapping",
                    "license": _LICENSE,
                }

        parser = _ConfiguredParser(version="245")
        ms = parser.create_mapping_set([], mapping_type="id")
        assert ms.mapping_set_title == "HGNC Mapping (245)"


class TestResolveVersionFromSparql:
    """_resolve_version_from_sparql: version_query results feed version/release_date."""

    def test_release_date_is_a_plain_date_not_the_raw_datetime_literal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """release_date must be YYYY-MM-DD, or SSSOM's XSDDate rejects it as mapping_date.

        Regression test: a live SPARQL endpoint's dateModified literal
        (e.g. Wikidata's dump date) comes back as a full datetime with a
        time and "Z" suffix -- storing that raw string in release_date
        crashed create_mapping_set with "not a valid date" in production.
        """
        import mapkgsutils.parsers.base as base_module

        monkeypatch.setattr(
            base_module, "query_sparql_scalar", lambda query, endpoint: "2026-07-10T23:32:31Z"
        )

        parser = _ToyParser()
        parser._config = DatasourceConfig(
            name="Wikidata",
            prefix="WD",
            curie_base_url="http://www.wikidata.org/entity/",
            sparql_endpoint="https://qlever.dev/api/wikidata",
            version_query="SELECT ?date WHERE { ?s ?p ?date }",
        )

        assert parser._resolve_version() == "2026-07-10"
        assert parser.release_date == "2026-07-10"
        assert parser._resolve_mapping_date() == "2026-07-10"

    def test_unconfigured_datasource_is_a_no_op(self) -> None:
        """No version_query/sparql_endpoint: _resolve_version_from_sparql returns None."""
        assert _ToyParser()._resolve_version_from_sparql() is None
