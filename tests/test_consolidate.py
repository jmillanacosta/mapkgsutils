"""Tests for the generic mapping-date consolidation cache and walk skeleton."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from sssom_schema import Mapping

from mapkgsutils.consolidate import (
    build_consolidated_mapping_set,
    consolidate_by_date,
    consolidate_by_release,
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
        """No cache yet: an empty dict, not an error."""
        assert read_cache(tmp_path / "missing.tsv") == {}

    def test_write_then_read_meta_round_trips(self, tmp_path: Path) -> None:
        """The last_version sidecar comes back as the same string."""
        meta_path = tmp_path / "meta.json"
        write_meta(meta_path, "245")
        assert read_meta(meta_path) == "245"

    def test_read_meta_missing_file_returns_none(self, tmp_path: Path) -> None:
        """No sidecar yet: None, not an error."""
        assert read_meta(tmp_path / "missing.json") is None


class TestLoadMappingDates:
    """load_mapping_dates exposes only records with a real resolved date."""

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

    def test_missing_cache_returns_empty_dict(self, tmp_path: Path) -> None:
        """Safe to call before anything has been consolidated."""
        assert load_mapping_dates(tmp_path / "missing.tsv") == {}


class TestConsolidateByDate:
    """Single-pass "date" mode: capture each row's own real mapping_date."""

    def test_returns_false_and_writes_nothing_when_no_dates(self, tmp_path: Path) -> None:
        """A mapping set with no per-row dates can't seed the cache."""
        cache_path = tmp_path / "cache.tsv"
        meta_path = tmp_path / "meta.json"
        mapping_set = SimpleNamespace(mappings=[_mapping("SEC:1", "PRI:1", "ns/" + "a" * 16)])

        got_dates = consolidate_by_date(cache_path, meta_path, run_one_version=lambda: mapping_set)

        assert got_dates is False
        assert not cache_path.exists()

    def test_captures_real_per_row_dates(self, tmp_path: Path) -> None:
        """Dated rows are keyed by their trailing pair hash and written to the cache."""
        cache_path = tmp_path / "cache.tsv"
        meta_path = tmp_path / "meta.json"
        mapping_set = SimpleNamespace(
            mapping_set_version="2026-01-01",
            mappings=[
                _mapping("SEC:1", "PRI:1", "ns/" + "a" * 16, mapping_date="2020-05-01"),
            ],
        )

        got_dates = consolidate_by_date(cache_path, meta_path, run_one_version=lambda: mapping_set)

        assert got_dates is True
        records = read_cache(cache_path)
        assert records["a" * 16]["first_seen_date"] == "2020-05-01"
        assert read_meta(meta_path) == "2026-01-01"


class TestConsolidateByRelease:
    """Historical-walk "release" mode, fully in-memory (no network)."""

    def test_tracks_first_and_last_seen_across_versions(self, tmp_path: Path) -> None:
        """A mapping's first_seen is its earliest version; last_seen keeps bumping."""
        cache_path = tmp_path / "cache.tsv"
        meta_path = tmp_path / "meta.json"
        by_version = {
            "1": [_mapping("SEC:1", "PRI:1", "1/" + "a" * 16)],
            "2": [_mapping("SEC:1", "PRI:1", "2/" + "a" * 16)],
        }

        consolidate_by_release(
            cache_path,
            meta_path,
            label="test",
            list_versions=lambda: ["1", "2"],
            run_one_version=lambda v: SimpleNamespace(mappings=by_version[v]),
            resolve_release_date=lambda v: None,
            show_progress=False,
        )

        records = read_cache(cache_path)
        assert records["a" * 16]["first_seen_version"] == "1"
        assert records["a" * 16]["last_seen_version"] == "2"
        # No resolvable release date: the version label isn't stored as a date.
        assert records["a" * 16]["first_seen_date"] == ""
        assert read_meta(meta_path) == "2"

    def test_resumes_from_last_version_unless_forced(self, tmp_path: Path) -> None:
        """A second run only walks versions past the resumed last_version."""
        cache_path = tmp_path / "cache.tsv"
        meta_path = tmp_path / "meta.json"
        seen: list[str] = []

        def _run(v: str) -> SimpleNamespace:
            seen.append(v)
            return SimpleNamespace(mappings=[])

        consolidate_by_release(
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
        consolidate_by_release(
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
        consolidate_by_release(
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

        def _run(v: str) -> SimpleNamespace:
            if v == "2":
                raise ValueError("simulated failure")
            return SimpleNamespace(mappings=[_mapping("SEC:1", "PRI:1", f"{v}/" + "a" * 16)])

        consolidate_by_release(
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

    output_path = write_consolidated_sssom(
        cache_path,
        meta_path,
        mapping_set_class=BaseMappingSet,
        record_namespace="ns/",
        mapping_set_metadata=_METADATA,
    )

    assert output_path.exists()
    assert "SEC:1" in output_path.read_text(encoding="utf-8")
