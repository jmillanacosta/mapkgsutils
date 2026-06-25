"""Diff operations for comparing MappingSets between releases.

Uses Polars for efficient comparison of large mapping datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from sssom_schema import MappingSet

__all__ = [
    "MappingDiff",
    "diff_mapping_sets",
    "diff_sssom_files",
    "summarize_diff",
]


@dataclass
class MappingDiff:
    """Result of comparing two MappingSets."""

    old_version: str | None
    new_version: str | None
    datasource: str

    # Added mappings (object_id, subject_id) - in new but not in old
    added: pl.DataFrame = field(default_factory=pl.DataFrame)
    # Removed mappings (object_id, subject_id) - in old but not in new
    removed: pl.DataFrame = field(default_factory=pl.DataFrame)
    # Changed mappings (same object_id, different subject_id)
    changed: pl.DataFrame = field(default_factory=pl.DataFrame)
    # Intersection mappings (in both old and new)
    intersection: pl.DataFrame = field(default_factory=pl.DataFrame)

    @property
    def added_count(self) -> int:
        """Number of added mappings."""
        return len(self.added)

    @property
    def removed_count(self) -> int:
        """Number of removed mappings."""
        return len(self.removed)

    @property
    def changed_count(self) -> int:
        """Number of changed mappings."""
        return len(self.changed)

    @property
    def intersection_count(self) -> int:
        """Number of mappings in both sets."""
        return len(self.intersection)

    @property
    def total_changes(self) -> int:
        """Total number of changes."""
        return self.added_count + self.removed_count + self.changed_count

    @property
    def has_changes(self) -> bool:
        """Whether there are any changes."""
        return self.total_changes > 0


def mapping_set_to_dataframe(mapping_set: MappingSet) -> pl.DataFrame:
    """Convert a MappingSet to a Polars DataFrame for comparison.

    Args:
        mapping_set: The MappingSet to convert (sssom_schema.MappingSet).

    Returns:
        Polars DataFrame with columns: subject_id, object_id.
    """
    rows = []
    mappings = mapping_set.mappings or []
    for mapping in mappings:
        rows.append(
            {
                "subject_id": mapping.subject_id,
                "object_id": mapping.object_id,
            }
        )

    if not rows:
        return pl.DataFrame(schema={"subject_id": pl.Utf8, "object_id": pl.Utf8})

    return pl.DataFrame(rows)


def diff_mapping_sets(
    old_set: MappingSet,
    new_set: MappingSet,
    datasource: str = "unknown",
) -> MappingDiff:
    """Compare two MappingSets and find differences.

    Args:
        old_set: The older/previous MappingSet (sssom_schema.MappingSet).
        new_set: The newer/current MappingSet (sssom_schema.MappingSet).
        datasource: Name of the datasource for the diff.

    Returns:
        MappingDiff with added, removed, and changed mappings.
    """
    old_df = mapping_set_to_dataframe(old_set)
    new_df = mapping_set_to_dataframe(new_set)

    return _diff_dataframes(
        old_df=old_df,
        new_df=new_df,
        old_version=old_set.mapping_set_version,
        new_version=new_set.mapping_set_version,
        datasource=datasource,
    )


def diff_sssom_files(
    old_file: Path | str,
    new_file: Path | str,
    datasource: str = "unknown",
) -> MappingDiff:
    """Compare two SSSOM TSV files and find differences.

    Args:
        old_file: Path to the older SSSOM file.
        new_file: Path to the newer SSSOM file.
        datasource: Name of the datasource.

    Returns:
        MappingDiff with added, removed, and changed mappings.
    """
    old_path = Path(old_file)
    new_path = Path(new_file)

    # Read SSSOM files (skip metadata lines starting with #)
    old_df = _read_sssom_to_dataframe(old_path)
    new_df = _read_sssom_to_dataframe(new_path)

    # Extract versions from metadata if available
    old_version = _extract_sssom_version(old_path)
    new_version = _extract_sssom_version(new_path)

    return _diff_dataframes(
        old_df=old_df,
        new_df=new_df,
        old_version=old_version,
        new_version=new_version,
        datasource=datasource,
    )


def _read_sssom_to_dataframe(path: Path) -> pl.DataFrame:
    """Read an SSSOM TSV file to a Polars DataFrame.

    Handles the SSSOM metadata header (lines starting with #).

    Args:
        path: Path to the SSSOM file.

    Returns:
        Polars DataFrame with subject_id and object_id columns.
    """
    # Count header lines
    header_lines = 0
    with path.open() as f:
        for line in f:
            if line.startswith("#"):
                header_lines += 1
            else:
                break

    # Read the TSV, skipping metadata
    df = pl.read_csv(
        path,
        separator="\t",
        skip_rows=header_lines,
        infer_schema_length=10000,
    )

    if "subject_id" in df.columns and "object_id" in df.columns:
        df = df.select(["subject_id", "object_id"])

    return df


def _extract_sssom_version(path: Path) -> str | None:
    """Extract version from SSSOM metadata header.

    Args:
        path: Path to the SSSOM file.

    Returns:
        Version string or None if not found.
    """
    with path.open() as f:
        for line in f:
            if not line.startswith("#"):
                break
            if "mapping_set_version" in line.lower():
                # Format: #mapping_set_version: "1.0"
                parts = line.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip().strip('"').strip("'")
    return None


def _diff_dataframes(
    old_df: pl.DataFrame,
    new_df: pl.DataFrame,
    old_version: str | None,
    new_version: str | None,
    datasource: str,
) -> MappingDiff:
    """Compare two DataFrames and find differences.

    Args:
        old_df: The older DataFrame.
        new_df: The newer DataFrame.
        old_version: Version of the old set.
        new_version: Version of the new set.
        datasource: Datasource name.

    Returns:
        MappingDiff with added, removed, and changed mappings.
    """
    # Ensure consistent schema
    if old_df.is_empty():
        old_df = pl.DataFrame(schema={"subject_id": pl.Utf8, "object_id": pl.Utf8})
    if new_df.is_empty():
        new_df = pl.DataFrame(schema={"subject_id": pl.Utf8, "object_id": pl.Utf8})

    # Find mappings by (subject_id, object_id) pairs
    old_pairs = old_df.select(["subject_id", "object_id"]).unique()
    new_pairs = new_df.select(["subject_id", "object_id"]).unique()

    # Added: in new but not in old
    added = new_pairs.join(
        old_pairs,
        on=["subject_id", "object_id"],
        how="anti",
    )

    # Removed: in old but not in new
    removed = old_pairs.join(
        new_pairs,
        on=["subject_id", "object_id"],
        how="anti",
    )

    # Changed: same object_id but different subject_id
    # Get unique secondary IDs with their primary mappings
    old_by_sec = old_df.select(
        [
            pl.col("object_id"),
            pl.col("subject_id").alias("old_subject_id"),
        ]
    ).unique()

    new_by_sec = new_df.select(
        [
            pl.col("object_id"),
            pl.col("subject_id").alias("new_subject_id"),
        ]
    ).unique()

    # Join on object_id and filter where primary changed
    changed = old_by_sec.join(new_by_sec, on="object_id", how="inner").filter(
        pl.col("old_subject_id") != pl.col("new_subject_id")
    )

    # Intersection: in both old and new (same subject_id, object_id pair)
    intersection = old_pairs.join(
        new_pairs,
        on=["subject_id", "object_id"],
        how="inner",
    )

    return MappingDiff(
        old_version=old_version,
        new_version=new_version,
        datasource=datasource,
        added=added,
        removed=removed,
        changed=changed,
        intersection=intersection,
    )


def summarize_diff(diff: MappingDiff) -> str:
    """Generate a human-readable summary of a diff.

    Args:
        diff: The MappingDiff to summarize.

    Returns:
        A formatted string summary.
    """
    lines = [
        f"Diff Summary for {diff.datasource}",
        f"  Old version: {diff.old_version or 'unknown'}",
        f"  New version: {diff.new_version or 'unknown'}",
        "",
        f"  Intersection:     {diff.intersection_count:>8}",
        f"  Added mappings:   {diff.added_count:>8}",
        f"  Removed mappings: {diff.removed_count:>8}",
        f"  Changed mappings: {diff.changed_count:>8}",
        f"  Total changes:    {diff.total_changes:>8}",
    ]

    if diff.added_count > 0 and diff.added_count <= 10:
        lines.append("")
        lines.append("  Added:")
        for row in diff.added.iter_rows(named=True):
            lines.append(f"    {row['object_id']} -> {row['subject_id']}")

    if diff.removed_count > 0 and diff.removed_count <= 10:
        lines.append("")
        lines.append("  Removed:")
        for row in diff.removed.iter_rows(named=True):
            lines.append(f"    {row['object_id']} -> {row['subject_id']}")

    if diff.changed_count > 0 and diff.changed_count <= 10:
        lines.append("")
        lines.append("  Changed:")
        for row in diff.changed.iter_rows(named=True):
            lines.append(
                f"    {row['object_id']}: {row['old_subject_id']} -> {row['new_subject_id']}"
            )

    return "\n".join(lines)
