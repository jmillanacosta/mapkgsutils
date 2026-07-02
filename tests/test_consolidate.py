"""Tests for the generic mapping-date consolidation cache and walk skeleton."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from sssom_schema import Mapping

from mapkgsutils.consolidate import (
    build_consolidated_mapping_set,
    consolidate,
    load_mapping_dates,
    read_cache,
    read_meta,
    write_cache,
    write_consolidated_sssom,
    write_meta,
)
from mapkgsutils.parsers.base import BaseMappingSet

_LICENSE = "https://creativecommons.org/publicdomain/zero/1.0/"
_METADATA = {"mapping_set_id": "https://example.org/test", "license": _LICENSE}


def _mapping(subject_id: str, object_id: str, record_id: str, **kwargs: object) -> Mapping:
    return Mapping(
        subject_id=subject_id,
        object_id=object_id,
        record_id=record_id,
        predicate_id="IAO:0100001",
        mapping_justification="semapv:BackgroundKnowledgeBasedMatching",
        **kwargs,
    )


class TestCacheIO:
    """read_cache/write_cache and read_meta/write_meta round-trip through a TSV/JSON file."""

    def test_write_then_read_cache_round_trips(self, tmp_path: Path) -> None:
        """Every field written for a record_id comes back unchanged."""
        cache_path = tmp_path / "cache.tsv"
        records = {
            "abc123": {
                "first_seen_version": "1",
                "first_seen_date": "2020-01-01",
                "last_seen_version": "3",
                "last_seen_date": "2020-03-01",
            }
        }
        write_cache(cache_path, records)
        assert read_cache(cache_path) == {
            "abc123": {
                "first_seen_version": "1",
                "first_seen_date": "2020-01-01",
                "last_seen_version": "3",
                "last_seen_date": "2020-03-01",
                "fields_json": "",
            }
        }

    def test_read_cache_missing_file_returns_empty_dict(self, tmp_path: Path) -> None:
        """No cache yet: an empty dict."""
        assert read_cache(tmp_path / "missing.tsv") == {}

    def test_write_then_read_meta_round_trips(self, tmp_path: Path) -> None:
        """The last_version sidecar comes back as the same string."""
        meta_path = tmp_path / "meta.json"
        write_meta(meta_path, "245")
        assert read_meta(meta_path) == "245"

    def test_read_meta_missing_file_returns_none(self, tmp_path: Path) -> None:
        """No meta yet: None."""
        assert read_meta(tmp_path / "missing.json") is None


class TestLoadMappingDates:
    """load_mapping_dates exposes the best available date per record."""

    def test_omits_records_with_no_resolved_date(self, tmp_path: Path) -> None:
        """A record walked but never assigned a date doesn't come back at all."""
        cache_path = tmp_path / "cache.tsv"
        write_cache(
            cache_path,
            {
                "dated": {"first_seen_version": "245", "first_seen_date": "2020-01-01"},
                "undated": {"first_seen_version": "183", "first_seen_date": ""},
            },
        )
        assert load_mapping_dates(cache_path) == {"dated": "2020-01-01"}

    def test_prefers_per_row_date_over_first_seen_release_date(self, tmp_path: Path) -> None:
        """The mapping date (in fields_json) wins over the release date."""
        cache_path = tmp_path / "cache.tsv"
        fields_json = json.dumps(
            {
                "subject_id": "SEC:1",
                "object_id": "PRI:1",
                "predicate_id": "IAO:0100001",
                "mapping_justification": "semapv:BackgroundKnowledgeBasedMatching",
                "mapping_date": "2019-03-03",
            }
        )
        write_cache(
            cache_path,
            {
                "a" * 16: {
                    "first_seen_version": "100",
                    "first_seen_date": "2020-04-28",
                    "fields_json": fields_json,
                }
            },
        )
        assert load_mapping_dates(cache_path) == {"a" * 16: "2019-03-03"}

    def test_missing_cache_returns_empty_dict(self, tmp_path: Path) -> None:
        """Safe to call before anything has been consolidated."""
        assert load_mapping_dates(tmp_path / "missing.tsv") == {}


class TestConsolidateSingleParse:
    """No versioned archive (list_versions=None): a single parse keeping every mapping."""

    def test_single_parse_keeps_undated_rows(self, tmp_path: Path) -> None:
        """Every mapping is cached from one parse — undated rows are not dropped."""
        cache_path = tmp_path / "cache.tsv"
        meta_path = tmp_path / "meta.json"
        parses: list[str | None] = []

        def _run(v: str | None) -> SimpleNamespace:
            parses.append(v)
            return SimpleNamespace(
                mapping_set_version="2026-01-01",
                mappings=[
                    _mapping("SEC:1", "PRI:1", "cur/" + "a" * 16, mapping_date="2020-05-01"),
                    _mapping("SEC:2", "PRI:2", "cur/" + "b" * 16),  # undated
                ],
            )

        consolidate(cache_path, meta_path, label="test", run_one_version=_run, show_progress=False)

        assert parses == [None]  # a single parse, called with None
        records = read_cache(cache_path)
        assert set(records) == {"a" * 16, "b" * 16}  # undated kept alongside dated
        assert records["a" * 16]["first_seen_date"] == "2020-05-01"
        assert records["b" * 16]["first_seen_date"] == ""
        assert read_meta(meta_path) == "2026-01-01"


class TestConsolidateReleaseWalk:
    """Versioned archive (list_versions provided): the historical walk, fully in-memory."""

    def test_tracks_first_and_last_seen_across_versions(self, tmp_path: Path) -> None:
        """A mapping's first_seen is its earliest version; last_seen keeps bumping."""
        cache_path = tmp_path / "cache.tsv"
        meta_path = tmp_path / "meta.json"
        by_version = {
            "1": [_mapping("SEC:1", "PRI:1", "1/" + "a" * 16)],
            "2": [_mapping("SEC:1", "PRI:1", "2/" + "a" * 16)],
        }

        consolidate(
            cache_path,
            meta_path,
            label="test",
            list_versions=lambda: ["1", "2"],
            run_one_version=lambda v: SimpleNamespace(mappings=by_version[str(v)]),
            resolve_release_date=lambda v: None,
            show_progress=False,
        )

        records = read_cache(cache_path)
        assert records["a" * 16]["first_seen_version"] == "1"
        assert records["a" * 16]["last_seen_version"] == "2"
        # No resolvable release date: the version label isn't stored as a date.
        assert records["a" * 16]["first_seen_date"] == ""
        assert read_meta(meta_path) == "2"

    def test_per_row_mapping_date_survives_the_walk(self, tmp_path: Path) -> None:
        """A mapping's own date stays in mapping_date; the version fields hold the release."""
        cache_path = tmp_path / "cache.tsv"
        meta_path = tmp_path / "meta.json"
        m = _mapping("SEC:1", "PRI:1", "1/" + "a" * 16, mapping_date="2019-03-03")

        consolidate(
            cache_path,
            meta_path,
            label="test",
            list_versions=lambda: ["1"],
            run_one_version=lambda v: SimpleNamespace(mappings=[m]),
            # A resolvable *release* date, distinct from the per-row mapping_date.
            resolve_release_date=lambda v: datetime(2020, 4, 28),
            show_progress=False,
        )

        mapping_set = build_consolidated_mapping_set(
            read_cache(cache_path),
            read_meta(meta_path),
            mapping_set_class=BaseMappingSet,
            record_namespace="ns/",
            mapping_set_metadata=_METADATA,
        )
        built = mapping_set.mappings[0]
        assert str(built.mapping_date) == "2019-03-03"  # actual date it appeared
        assert built.subject_source_version == "1"  # first-seen release version
        assert built.object_source_version == "1"

    def test_resumes_from_last_version_unless_forced(self, tmp_path: Path) -> None:
        """A second run only walks versions past the resumed last_version."""
        cache_path = tmp_path / "cache.tsv"
        meta_path = tmp_path / "meta.json"
        seen: list[str | None] = []

        def _run(v: str | None) -> SimpleNamespace:
            seen.append(v)
            return SimpleNamespace(mappings=[])

        consolidate(
            cache_path,
            meta_path,
            label="test",
            list_versions=lambda: ["1", "2"],
            run_one_version=_run,
            resolve_release_date=lambda v: None,
            show_progress=False,
        )
        assert seen == ["1", "2"]

        seen.clear()
        consolidate(
            cache_path,
            meta_path,
            label="test",
            list_versions=lambda: ["1", "2"],
            run_one_version=_run,
            resolve_release_date=lambda v: None,
            show_progress=False,
        )
        assert seen == []

        seen.clear()
        consolidate(
            cache_path,
            meta_path,
            label="test",
            list_versions=lambda: ["1", "2"],
            run_one_version=_run,
            resolve_release_date=lambda v: None,
            show_progress=False,
            force=True,
        )
        assert seen == ["1", "2"]

    def test_a_failing_version_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        """An exception for one version is swallowed; later versions still merge."""
        cache_path = tmp_path / "cache.tsv"
        meta_path = tmp_path / "meta.json"

        def _run(v: str | None) -> SimpleNamespace:
            if v == "2":
                raise ValueError("simulated failure")
            return SimpleNamespace(mappings=[_mapping("SEC:1", "PRI:1", f"{v}/" + "a" * 16)])

        consolidate(
            cache_path,
            meta_path,
            label="test",
            list_versions=lambda: ["1", "2", "3"],
            run_one_version=_run,
            resolve_release_date=lambda v: None,
            show_progress=False,
        )

        records = read_cache(cache_path)
        assert records["a" * 16]["last_seen_version"] == "3"
        assert read_meta(meta_path) == "3"


class TestBuildConsolidatedMappingSet:
    """Materializing the cache back into a real SSSOM mapping set."""

    def test_rebuilds_mappings_with_record_id_scoped_to_consolidate(self) -> None:
        """Each row's record_id is rebuilt under .../consolidate/<pair_hash>."""
        fields_json = json.dumps(
            {
                "subject_id": "SEC:1",
                "object_id": "PRI:1",
                "predicate_id": "IAO:0100001",
                "mapping_justification": "semapv:BackgroundKnowledgeBasedMatching",
            }
        )
        records = {
            "a" * 16: {
                "first_seen_version": "183",
                "first_seen_date": "2020-01-01",
                "fields_json": fields_json,
            }
        }

        mapping_set = build_consolidated_mapping_set(
            records,
            "245",
            mapping_set_class=BaseMappingSet,
            record_namespace="ns/",
            mapping_set_metadata=_METADATA,
        )

        assert len(mapping_set.mappings) == 1
        m = mapping_set.mappings[0]
        assert m.record_id == "ns/245/consolidate/" + "a" * 16
        assert str(m.mapping_date) == "2020-01-01"

    def test_version_and_date_go_to_their_own_sssom_fields(self) -> None:
        """A release version never lands in mapping_date, and vice versa."""
        fields_json = json.dumps(
            {
                "subject_id": "SEC:1",
                "object_id": "PRI:1",
                "predicate_id": "IAO:0100001",
                "mapping_justification": "semapv:BackgroundKnowledgeBasedMatching",
            }
        )
        records = {
            "a" * 16: {
                "first_seen_version": "183",
                "first_seen_date": "2020-01-01",
                "fields_json": fields_json,
            }
        }

        mapping_set = build_consolidated_mapping_set(
            records,
            "245",
            mapping_set_class=BaseMappingSet,
            record_namespace="ns/",
            mapping_set_metadata=_METADATA,
        )

        m = mapping_set.mappings[0]
        assert str(m.mapping_date) == "2020-01-01"
        assert m.subject_source_version == "183"
        assert m.object_source_version == "183"

    def test_per_row_date_wins_over_first_seen_release_date(self) -> None:
        """When the snapshot carries its own mapping_date, that date is kept."""
        fields_json = json.dumps(
            {
                "subject_id": "SEC:1",
                "object_id": "PRI:1",
                "predicate_id": "IAO:0100001",
                "mapping_justification": "semapv:BackgroundKnowledgeBasedMatching",
                "mapping_date": "2019-03-03",
            }
        )
        records = {
            "a" * 16: {
                "first_seen_version": "183",
                "first_seen_date": "2020-01-01",
                "fields_json": fields_json,
            }
        }

        mapping_set = build_consolidated_mapping_set(
            records,
            "245",
            mapping_set_class=BaseMappingSet,
            record_namespace="ns/",
            mapping_set_metadata=_METADATA,
        )

        m = mapping_set.mappings[0]
        assert str(m.mapping_date) == "2019-03-03"  # the per-row date, not first_seen_date
        assert m.subject_source_version == "183"
        assert m.object_source_version == "183"

    def test_no_resolved_date_still_records_the_version(self) -> None:
        """No real date for this row, but the release it first appeared in is kept."""
        fields_json = json.dumps(
            {
                "subject_id": "SEC:1",
                "object_id": "PRI:1",
                "predicate_id": "IAO:0100001",
                "mapping_justification": "semapv:BackgroundKnowledgeBasedMatching",
            }
        )
        records = {
            "a" * 16: {
                "first_seen_version": "183",
                "first_seen_date": "",
                "fields_json": fields_json,
            }
        }

        mapping_set = build_consolidated_mapping_set(
            records,
            "245",
            mapping_set_class=BaseMappingSet,
            record_namespace="ns/",
            mapping_set_metadata=_METADATA,
        )

        m = mapping_set.mappings[0]
        assert m.mapping_date is None
        assert m.subject_source_version == "183"
        assert m.object_source_version == "183"

    def test_empty_first_seen_date_leaves_mapping_date_unset(self) -> None:
        """An unresolved date doesn't get passed into Mapping(mapping_date=...)."""
        fields_json = json.dumps(
            {
                "subject_id": "SEC:1",
                "object_id": "PRI:1",
                "predicate_id": "IAO:0100001",
                "mapping_justification": "semapv:BackgroundKnowledgeBasedMatching",
            }
        )
        records = {"a" * 16: {"first_seen_date": "", "fields_json": fields_json}}

        mapping_set = build_consolidated_mapping_set(
            records,
            "183",
            mapping_set_class=BaseMappingSet,
            record_namespace="ns/",
            mapping_set_metadata=_METADATA,
        )

        assert mapping_set.mappings[0].mapping_date is None

    def test_rows_with_no_fields_json_are_skipped(self) -> None:
        """Legacy/hand-built cache rows with no field snapshot aren't materialized."""
        records = {"a" * 16: {"first_seen_date": "2020-01-01", "fields_json": ""}}

        mapping_set = build_consolidated_mapping_set(
            records,
            "245",
            mapping_set_class=BaseMappingSet,
            record_namespace="ns/",
            mapping_set_metadata=_METADATA,
        )

        assert mapping_set.mappings == []


def test_write_consolidated_sssom_writes_a_companion_file(tmp_path: Path) -> None:
    """write_consolidated_sssom reads the cache and writes a real SSSOM TSV next to it."""
    cache_path = tmp_path / "cache.tsv"
    meta_path = tmp_path / "meta.json"
    fields_json = json.dumps(
        {
            "subject_id": "SEC:1",
            "object_id": "PRI:1",
            "predicate_id": "IAO:0100001",
            "mapping_justification": "semapv:BackgroundKnowledgeBasedMatching",
        }
    )
    write_cache(
        cache_path, {"a" * 16: {"first_seen_date": "2020-01-01", "fields_json": fields_json}}
    )
    write_meta(meta_path, "245")

    output_path, mapping_set = write_consolidated_sssom(
        cache_path,
        meta_path,
        mapping_set_class=BaseMappingSet,
        record_namespace="ns/",
        mapping_set_metadata=_METADATA,
    )

    assert output_path.exists()
    assert "SEC:1" in output_path.read_text(encoding="utf-8")
    assert mapping_set.mappings


def test_uniform_provenance_still_emits_columns(tmp_path: Path) -> None:
    """Regression (Bug 1): rows sharing one first-seen version/date keep the columns.

    When every mapping first appeared in the same release, first_seen_version/
    first_seen_date are uniform across all rows. With sssom's default
    condense=True those slots would be lifted into the YAML header and vanish
    as columns; the fix writes them as per-row columns regardless.
    """
    cache_path = tmp_path / "cache.tsv"
    meta_path = tmp_path / "meta.json"
    records = {}
    for i in range(5):
        fields_json = json.dumps(
            {
                "subject_id": f"CHEBI:{i}",
                "object_id": f"CHEBI:{i + 1}",
                "predicate_id": "IAO:0100001",
                "mapping_justification": "semapv:BackgroundKnowledgeBasedMatching",
            }
        )
        records[f"{i:016d}"] = {
            "first_seen_version": "100",
            "first_seen_date": "2020-04-28",
            "last_seen_version": "116",
            "last_seen_date": "2024-08-28",
            "fields_json": fields_json,
        }
    write_cache(cache_path, records)
    write_meta(meta_path, "116")

    output_path, _ = write_consolidated_sssom(
        cache_path,
        meta_path,
        mapping_set_class=BaseMappingSet,
        record_namespace="ns/",
        mapping_set_metadata=_METADATA,
    )

    text = output_path.read_text(encoding="utf-8")
    data_lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    header_cols = data_lines[0].split("\t")
    assert "mapping_date" in header_cols
    assert "subject_source_version" in header_cols
    assert "object_source_version" in header_cols
