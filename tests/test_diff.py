"""Tests for the diff module: MappingDiff, diff_mapping_sets, diff_sssom_files, summarize_diff."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pytest
from sssom_schema import Mapping, MappingSet

from mapkgsutils.diff import (
    MappingDiff,
    diff_mapping_sets,
    diff_sssom_files,
    mapping_set_to_dataframe,
    summarize_diff,
)


def _mapping_set(mapping_set_id: str, version: str, pairs: list[tuple[str, str]]) -> MappingSet:
    ms = MappingSet(mapping_set_id=mapping_set_id, license="https://example.org/license")
    ms.mapping_set_version = version
    ms.mappings = [
        Mapping(
            subject_id=pri,
            predicate_id="IAO:0100001",
            object_id=sec,
            mapping_justification="semapv:BackgroundKnowledgeBasedMatching",
        )
        for pri, sec in pairs
    ]
    return ms


@pytest.fixture
def v1() -> MappingSet:
    """Three mappings: SECONDARY1/2/3 -> PRIMARY1/2/3."""
    return _mapping_set(
        "test:v1",
        "1.0",
        [("PRIMARY1", "SECONDARY1"), ("PRIMARY2", "SECONDARY2"), ("PRIMARY3", "SECONDARY3")],
    )


@pytest.fixture
def v2() -> MappingSet:
    """Return the v1 set with one row changed, one removed, and one added."""
    return _mapping_set(
        "test:v2",
        "2.0",
        [("PRIMARY1", "SECONDARY1"), ("PRIMARY4", "SECONDARY2"), ("PRIMARY5", "SECONDARY5")],
    )


def test_mapping_diff_properties() -> None:
    """added/removed/changed counts, total_changes, and has_changes all derive from the frames."""
    empty = MappingDiff(old_version="1.0", new_version="2.0", datasource="Test")
    assert (empty.added_count, empty.removed_count, empty.changed_count) == (0, 0, 0)
    assert empty.has_changes is False

    diff = MappingDiff(
        old_version="1.0",
        new_version="2.0",
        datasource="Test",
        added=pl.DataFrame({"subject_id": ["A"], "object_id": ["X"]}),
        removed=pl.DataFrame({"subject_id": ["B"], "object_id": ["Y"]}),
        changed=pl.DataFrame(
            {"object_id": ["Z"], "old_subject_id": ["C"], "new_subject_id": ["D"]}
        ),
    )
    assert (diff.added_count, diff.removed_count, diff.changed_count) == (1, 1, 1)
    assert diff.total_changes == 3
    assert diff.has_changes is True


def test_mapping_set_to_dataframe(v1: MappingSet) -> None:
    """Converts an empty and a populated MappingSet to a polars frame with the expected columns."""
    empty_df = mapping_set_to_dataframe(MappingSet(mapping_set_id="empty", license="x"))
    assert len(empty_df) == 0
    assert {"subject_id", "object_id"} <= set(empty_df.columns)

    df = mapping_set_to_dataframe(v1)
    assert len(df) == 3
    assert set(df["subject_id"]) == {"PRIMARY1", "PRIMARY2", "PRIMARY3"}


def test_diff_mapping_sets_identical_has_no_changes(v1: MappingSet) -> None:
    """Diffing a set against itself reports no changes."""
    diff = diff_mapping_sets(v1, v1)
    assert diff.has_changes is False


def test_diff_mapping_sets_detects_added_removed_and_changed(
    v1: MappingSet, v2: MappingSet
) -> None:
    """Added/removed/changed rows are classified correctly; version/datasource pass through."""
    diff = diff_mapping_sets(v1, v2, datasource="TestSource")
    assert diff.datasource == "TestSource"
    assert (diff.old_version, diff.new_version) == ("1.0", "2.0")
    # SECONDARY5 is new; SECONDARY3 is gone; SECONDARY2 now points at PRIMARY4
    assert "SECONDARY5" in diff.added["object_id"].to_list()
    assert "SECONDARY3" in diff.removed["object_id"].to_list()
    assert "SECONDARY2" in diff.changed["object_id"].to_list()


def test_diff_sssom_files() -> None:
    """File-based diffing reads the mapping_set_version from the SSSOM header, if present."""
    with TemporaryDirectory() as tmpdir:
        old_file = Path(tmpdir) / "old.tsv"
        new_file = Path(tmpdir) / "new.tsv"
        old_file.write_text(
            '#mapping_set_version: "1.0"\n'
            "subject_id\tobject_id\nPRIMARY1\tSECONDARY1\nPRIMARY2\tSECONDARY2\n"
        )
        new_file.write_text(
            '#mapping_set_version: "2.0"\n'
            "subject_id\tobject_id\nPRIMARY1\tSECONDARY1\nPRIMARY3\tSECONDARY3\n"
        )
        diff = diff_sssom_files(old_file, new_file, datasource="test")
        assert (diff.old_version, diff.new_version) == ("1.0", "2.0")
        assert diff.removed_count == 1
        assert diff.added_count == 1

        # No version header at all: both versions are None.
        new_file.write_text("subject_id\tobject_id\nPRIMARY1\tSECONDARY1\n")
        old_file.write_text("subject_id\tobject_id\nPRIMARY1\tSECONDARY1\n")
        no_version_diff = diff_sssom_files(old_file, new_file)
        assert (no_version_diff.old_version, no_version_diff.new_version) == (None, None)


def test_summarize_diff_includes_changes_and_truncates_long_lists() -> None:
    """The text summary names added/removed/changed entries, but truncates very long lists."""
    diff = MappingDiff(
        old_version="1.0",
        new_version="2.0",
        datasource="TestDB",
        added=pl.DataFrame({"subject_id": ["A"], "object_id": ["X"]}),
        removed=pl.DataFrame({"subject_id": ["B"], "object_id": ["Y"]}),
        changed=pl.DataFrame(
            {"object_id": ["SEC1"], "old_subject_id": ["OLD_PRI"], "new_subject_id": ["NEW_PRI"]}
        ),
    )
    summary = summarize_diff(diff)
    assert "TestDB" in summary
    assert "X" in summary and "Y" in summary  # added/removed objects shown
    assert "OLD_PRI" in summary and "NEW_PRI" in summary  # changed subjects shown

    big_diff = MappingDiff(
        old_version="1.0",
        new_version="2.0",
        datasource="TestDB",
        added=pl.DataFrame(
            {
                "subject_id": [f"PRI{i}" for i in range(20)],
                "object_id": [f"SEC{i}" for i in range(20)],
            }
        ),
    )
    big_summary = summarize_diff(big_diff)
    assert "20" in big_summary
    assert "SEC19" not in big_summary  # truncated after 10
