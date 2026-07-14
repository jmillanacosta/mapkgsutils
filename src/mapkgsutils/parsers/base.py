"""Base parser/downloader framework and mapping-set classes."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import fields as dataclass_fields
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast

import yaml
from sssom_schema import Mapping, MappingCardinalityEnum, MappingSet
from tqdm import tqdm

from mapkgsutils.logging import logger
from mapkgsutils.parsers.config import (
    DatasourceConfig,
    DistributionEra,
    XrefSource,
    get_datasource_config,
    load_config,
)
from mapkgsutils.parsers.config import _cmp_versions as _cmp_versions  # re-export for consolidate

if TYPE_CHECKING:
    import polars as pl
    import rdflib
    from sssom import sssom_document

_T = TypeVar("_T")

# Values for withdrawn entries

WITHDRAWN_ENTRY = "sssom:NoTermFound"
WITHDRAWN_ENTRY_LABEL = "Withdrawn Entry"


# Base Downloader Class


class BaseDownloader(ABC):
    """Abstract base class for datasource downloaders.

    Provides shared download logic that can be inherited by datasource-specific
    downloaders. Handles file downloads, URL construction, and version detection.
    """

    datasource_name: str = ""
    #: Importable package holding this datasource family's ``*.yaml``
    #: config files. Set by the concrete framework subclass (e.g.
    #: ``pysec2pri.parsers.base.BaseDownloader`` sets this to
    #: ``"pysec2pri.config"``); a downloader subclass never needs to.
    config_package: ClassVar[str] = ""
    _config: DatasourceConfig | None = None

    def __init__(
        self,
        version: str | None = None,
        show_progress: bool = True,
    ) -> None:
        """Initialize the downloader.

        Args:
            version: Version/release identifier for the datasource.
            show_progress: Whether to show progress bars during downloads.
        """
        self.version = version
        self.show_progress = show_progress

        # Load config from YAML
        if self.datasource_name:
            try:
                self._config = get_datasource_config(
                    self.datasource_name.lower(), config_package=self.config_package
                )
            except FileNotFoundError:
                self._config = None

    @property
    def config(self) -> DatasourceConfig | None:
        """Get the loaded configuration."""
        return self._config

    @property
    def new_format_version(self) -> int | None:
        """Get the version threshold for new format (if any)."""
        if self._config:
            return self._config.new_format_version
        return None

    def is_new_format(self, version: str | None = None) -> bool:
        """Check if a version uses the new format.

        Args:
            version: Version to check. If None, uses self.version.

        Returns:
            True if version >= new_format_version threshold.
        """
        v = version or self.version
        threshold = self.new_format_version

        if threshold is None:
            return True  # No threshold means always "new" format

        if v is None:
            return True  # Default to new format for latest

        try:
            return int(v) >= threshold
        except ValueError:
            return True  # Default to new if version is not numeric

    @abstractmethod
    def get_download_urls(
        self,
        version: str | None = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        """Get download URLs for the datasource.

        Args:
            version: Specific version to get URLs for.
            **kwargs: Additional options (e.g., subset, force_format).

        Returns:
            Dictionary mapping file keys to URLs.
        """

    @abstractmethod
    def download(
        self,
        output_dir: Path,
        version: str | None = None,
        decompress: bool = True,
        **kwargs: Any,
    ) -> dict[str, Path]:
        """Download files for the datasource.

        Args:
            output_dir: Directory to save downloaded files.
            version: Specific version to download.
            decompress: Whether to decompress .gz files.
            **kwargs: Additional options.

        Returns:
            Dictionary mapping file keys to downloaded paths.
        """

    def _download_file(
        self,
        url: str,
        output_path: Path,
        decompress_gz: bool = True,
        timeout: float | None = None,
        description: str | None = None,
    ) -> Path:
        """Download a file from URL to the specified path.

        Args:
            url: URL to download from.
            output_path: Where to save the file.
            decompress_gz: Whether to decompress .gz files automatically.
            timeout: Request timeout in seconds.
            description: Description for the progress bar.

        Returns:
            Path to the downloaded file.
        """
        from mapkgsutils.download import download_file

        return download_file(
            url,
            output_path,
            decompress_gz=decompress_gz,
            timeout=timeout,
            show_progress=self.show_progress,
            description=description,
        )

    def _download_urls(
        self,
        urls: dict[str, str],
        output_dir: Path,
        decompress: bool = True,
    ) -> dict[str, Path]:
        """Download files from URLs to output directory.

        Args:
            urls: Dictionary mapping file keys to URLs.
            output_dir: Directory to save files.
            decompress: Whether to decompress .gz files.

        Returns:
            Dictionary mapping file keys to downloaded paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded: dict[str, Path] = {}

        for key, url in urls.items():
            filename = url.split("/")[-1]

            if decompress and filename.endswith(".gz"):
                filename = filename[:-3]

            output_path = output_dir / filename
            logger.info("Downloading %s: %s", key, url)
            self._download_file(url, output_path, decompress_gz=decompress)
            downloaded[key] = output_path
            logger.info("Saved to: %s", output_path)

        return downloaded

    def list_versions(self) -> list[str]:
        """List all available archive versions for this datasource.

        Subclasses for datasources that publish versioned archives should
        override this method with source-specific retrieval logic.
        The base implementation raises :class:`ValueError`.

        Returns:
            Sorted list of version strings available for download.

        Raises:
            ValueError: Always, override in a subclass to provide versions.
        """
        name = self.datasource_name or type(self).__name__
        raise ValueError(
            f"{name.upper()} does not maintain a versioned archive. "
            "Only the latest release is available for download."
        )


def pair_hash(pri: str, sec: str) -> str:
    """Version-independent 16-hex-char digest for a (pri, sec) pair.

    The same pair always hashes identically, regardless of release/version
    or product (species/subset). A cross-release consolidation layer uses
    this as the join key to match a mapping across releases and discover
    when it first/last appeared. It is not what ends up in the
    ``record_id`` field; see :func:`mint_record_id` for that.
    """
    return hashlib.sha256(f"{pri}|{sec}".encode()).hexdigest()[:16]


def mint_record_id(pri: str, sec: str, *, namespace: str) -> str:
    """Mint a row's ``record_id``, the row's OWL Axiom IRI in SSSOM's RDF/OWL output.

    Scoped to *namespace* (typically a release- and product-specific prefix,
    see :meth:`BaseParser._record_namespace`), so the same (pri, sec) pair
    parsed from a different release/product gets a different ``record_id``.
    This matters because each :class:`~sssom_schema.Mapping` row is
    serialised as an ``owl:Axiom``: if record_id didn't vary across
    releases, loading several releases' SSSOM/RDF into one triplestore would
    assert contradictory axioms (different predicate, cardinality,
    confidence, ...) under the same IRI.

    The trailing 16 hex characters are always :func:`pair_hash`'s
    version-independent digest. Use that function directly, not this one,
    for cross-release matching/lookups.
    """
    return f"{namespace}{pair_hash(pri, sec)}"


class BaseMappingSet(MappingSet):  # type: ignore[misc]
    """A MappingSet with helpers for cardinality computation and export.

    Attributes:
        _primary_ids: Private store for the full primary ID set.
        _primary_labels: Private store for the full primary label set.
        _ambiguity_mode: Which field pair :func:`_find_ambiguous` checks for
            conflicts: ``"id"`` (subject_id/object_id) or ``"label"``
            (subject_label/object_label). Label-based subclasses override
            this to ``"label"``.
    """

    _ambiguity_mode: ClassVar[str] = "id"

    # Primaries are private to sssom's schema
    # Populated by parsers that have access to the full primary ID/label list
    # (e.g. an HGNC parser when the complete set file is provided).
    _primary_ids: set[str]
    # Maps label text to set of primary IDs that carry that label.
    _primary_labels: dict[str, set[str]]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the mapping set and the private primary-IDs store."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_primary_ids", set())
        object.__setattr__(self, "_primary_labels", {})

    # Export helpers

    def _default_stem(self) -> str:
        """Derive a base filename stem from mapping set metadata."""
        ms_id: str = str(getattr(self, "mapping_set_id", None) or "")
        if ms_id:
            stem = ms_id.rstrip("/").rsplit("/", 1)[-1]
        else:
            stem = str(getattr(self, "mapping_set_title", None) or "mapping_set")
            stem = stem.lower().replace(" ", "_")
        version = getattr(self, "mapping_set_version", None)
        if version:
            stem = f"{stem}_{version}"
        return stem

    def _resolve_path(self, output_path: Path | str | None, suffix: str) -> Path:
        """Return *output_path* if given, else auto-generate one."""
        if output_path is not None:
            return Path(output_path)
        return Path(f"{self._default_stem()}{suffix}")

    def to_sssom(self, output_path: Path | str | None = None) -> sssom_document.MappingSetDocument:
        """Return an SSSOM ``MappingSetDocument``, optionally writing to TSV.

        Args:
            output_path: If given, the document is also serialised to an SSSOM
                TSV file at this path

        Returns:
            :class:`sssom.sssom_document.MappingSetDocument` for the mapping set.
        """
        import curies
        from sssom.sssom_document import MappingSetDocument

        raw_curie_map: object = self.curie_map or {}
        records: list[curies.Record] = []
        if isinstance(raw_curie_map, dict):
            for k, v in raw_curie_map.items():
                if isinstance(v, str):
                    uri_prefix: str = v
                elif hasattr(v, "prefix_url"):
                    uri_prefix = cast(str, v.prefix_url)
                else:
                    continue
                records.append(curies.Record(prefix=k, uri_prefix=uri_prefix))
        converter = curies.Converter(records=records)
        doc = MappingSetDocument(mapping_set=self, converter=converter)

        if output_path is not None:
            from mapkgsutils.exports import write_sssom

            write_sssom(self, self._resolve_path(output_path, "_sssom.tsv"))

        return doc

    def to_rdf(
        self,
        output_path: Path | str | None = None,
        serialisation: str = "turtle",
    ) -> rdflib.Graph:
        """Return an RDFLib graph, optionally writing it to a file.

        When *output_path* is given (or auto-generated via the ``save``
        dispatcher), the graph is also serialised to disk.  Either way
        the :class:`rdflib.Graph` is returned so callers can query or
        manipulate it directly.

        Args:
            output_path: Destination path. Pass a path (or ``None`` to
                auto-generate one) to persist the graph.  If you only want
                the in-memory graph without touching the file-system, call
                ``to_rdf()`` with no arguments and ignore the path attribute.
            serialisation: RDFLib serialisation format (default: ``"turtle"``).

        Returns:
            :class:`rdflib.Graph` containing all mappings as RDF triples.
        """
        import io

        import rdflib
        from sssom.writers import write_rdf as _sssom_write_rdf

        from mapkgsutils.exports import _to_msdf_via_sssom_parser, write_rdf

        msdf = _to_msdf_via_sssom_parser(self)
        if msdf is None:
            raise ValueError("Failed to convert mapping set to RDF.")

        buf = io.StringIO()
        _sssom_write_rdf(msdf, buf, serialisation=serialisation)
        g = rdflib.Graph()
        g.parse(data=buf.getvalue(), format=serialisation)

        if output_path is not None:
            write_rdf(self, self._resolve_path(output_path, ".ttl"), serialisation=serialisation)

        return g

    def to_json(self, output_path: Path | str | None = None) -> dict[str, Any]:
        """Return the mapping set as a JSON-compatible ``dict``, optionally writing to file.

        Args:
            output_path: If given, the JSON is also written to this path.

        Returns:
            ``dict`` representation of the mapping set in SSSOM JSON format.
        """
        import io
        import json

        from sssom.writers import write_json as _sssom_write_json

        from mapkgsutils.exports import _to_msdf_via_sssom_parser

        msdf = _to_msdf_via_sssom_parser(self)
        if msdf is None:
            raise ValueError("Failed to convert mapping set to JSON.")
        buf = io.StringIO()
        _sssom_write_json(msdf, buf)
        data: dict[str, Any] = json.loads(buf.getvalue())

        if output_path is not None:
            path = self._resolve_path(output_path, ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(buf.getvalue(), encoding="utf-8")

        return data

    def to_owl(
        self, output_path: Path | str | None = None, serialisation: str = "turtle"
    ) -> rdflib.Graph:
        """Return an OWL ``rdflib.Graph``, optionally writing to file.

        Args:
            output_path: If given, the graph is also serialised to this path.
            serialisation: RDFLib serialisation format (default: ``"turtle"``).

        Returns:
            :class:`rdflib.Graph` containing OWL axioms for the mapping set.
        """
        import io

        import rdflib
        from sssom.writers import write_owl as _sssom_write_owl

        from mapkgsutils.exports import _to_msdf_via_sssom_parser

        msdf = _to_msdf_via_sssom_parser(self)
        if msdf is None:
            raise ValueError("Failed to convert mapping set to OWL.")
        buf = io.StringIO()
        _sssom_write_owl(msdf, buf, serialisation=serialisation)
        g = rdflib.Graph()
        g.parse(data=buf.getvalue(), format=serialisation)

        if output_path is not None:
            from mapkgsutils.exports import write_owl

            write_owl(
                self, self._resolve_path(output_path, "_owl.ttl"), serialisation=serialisation
            )

        return g

    def _save_shared(
        self,
        fmt: str,
        output_path: Path | str | None,
        **kwargs: object,
    ) -> Path | None:
        """Write one of the shared formats (sssom/rdf/json/owl).

        Returns the written :class:`Path`, or ``None`` if *fmt* is not a
        shared format (caller should handle it).
        """
        if fmt in ("rdf", "owl", "json"):
            from collections.abc import Callable as _Callable

            from mapkgsutils.exports import write_json, write_owl, write_rdf

            _write_fns: dict[str, tuple[_Callable[..., Path], str]] = {
                "rdf": (write_rdf, ".ttl"),
                "owl": (write_owl, "_owl.ttl"),
                "json": (write_json, ".json"),
            }
            fn, suffix = _write_fns[fmt]
            return fn(self, self._resolve_path(output_path, suffix), **kwargs)

        if fmt == "sssom":
            from mapkgsutils.exports import write_sssom

            return write_sssom(self, self._resolve_path(output_path, "_sssom.tsv"))

        return None

    def save(
        self,
        fmt: str,
        output_path: Path | str | None = None,
        **kwargs: object,
    ) -> Path:
        """Write to any supported format by name.

        Shared formats: ``"sssom"``, ``"rdf"``, ``"json"``, ``"owl"``.
        Subclasses override this to add type-specific formats.

        Args:
            fmt: Format key (see above).
            output_path: Destination path. Auto-generated if ``None``.
            **kwargs: Forwarded to the format-specific writer.

        Returns:
            Path to the written file.

        Raises:
            ValueError: For unknown format keys.
        """
        shared = self._save_shared(fmt, output_path, **kwargs)
        if shared is not None:
            return shared
        raise ValueError(f"Unknown format {fmt!r}. Choose from: json, owl, rdf, sssom")

    def find_ambiguous(self) -> AmbiguousMappingSet:
        """Find mappings whose subject is also a current primary entry.

        Delegates to :func:`_find_ambiguous`.  See that function for full
        semantics.

        Returns:
            :class:`AmbiguousMappingSet` with all conflicting mappings
            annotated.  Empty when no ambiguities are detected.
        """
        return _find_ambiguous(self)

    # Cardinality helpers

    def _compute_cardinalities(self, on: str = "id") -> None:
        """Compute and set mapping_cardinality on all mappings.

        'on' can be 'id' (uses subject_id/object_id) or 'label'.
        """
        if not self.mappings:  # type: ignore[has-type]
            return

        mappings = self._normalize_mappings()

        if on == "label":
            sec_field, pri_field = "subject_label", "object_label"
            sentinel = WITHDRAWN_ENTRY_LABEL
        else:
            sec_field, pri_field = "subject_id", "object_id"
            sentinel = WITHDRAWN_ENTRY

        import polars as pl

        sec_vals = [str(getattr(m, sec_field, None) or "") for m in mappings]
        pri_vals = [str(getattr(m, pri_field, None) or "") for m in mappings]

        df = pl.DataFrame({"sec": sec_vals, "pri": pri_vals})
        sec_is_nf = pl.col("sec") == sentinel
        pri_is_nf = pl.col("pri") == sentinel

        # Withdrawn (sssom:NoTermFound) rows are excluded from the
        # distinct-counterpart counts, matching sssom's own behavior.
        real = df.filter(~sec_is_nf & ~pri_is_nf)
        objects_per_subject = real.group_by("sec").agg(pl.col("pri").n_unique().alias("n_objects"))
        subjects_per_object = real.group_by("pri").agg(pl.col("sec").n_unique().alias("n_subjects"))

        cardinalities: list[str] = (
            df.join(objects_per_subject, on="sec", how="left", maintain_order="left")
            .join(subjects_per_object, on="pri", how="left", maintain_order="left")
            .select(
                pl.when(sec_is_nf & pri_is_nf)
                .then(pl.lit("0:0"))
                .when(sec_is_nf)
                .then(pl.lit("0:1"))
                .when(pri_is_nf)
                .then(pl.lit("1:0"))
                .when((pl.col("n_subjects") == 1) & (pl.col("n_objects") == 1))
                .then(pl.lit("1:1"))
                .when((pl.col("n_subjects") == 1) & (pl.col("n_objects") > 1))
                .then(pl.lit("1:n"))
                .when((pl.col("n_subjects") > 1) & (pl.col("n_objects") == 1))
                .then(pl.lit("n:1"))
                .otherwise(pl.lit("n:n"))
                .alias("cardinality")
            )
            .get_column("cardinality")
            .to_list()
        )

        for m, card in zip(mappings, cardinalities, strict=False):
            m.mapping_cardinality = MappingCardinalityEnum(card)

        self.mappings = mappings

    def _normalize_mappings(self) -> list[Mapping]:
        """Normalize mappings to a list of Mapping objects.

        Returns:
            List of Mapping objects.
        """
        mappings = self.mappings
        if not isinstance(mappings, list):
            mappings = [mappings]
        for i, m in enumerate(mappings):
            if isinstance(m, dict):
                mappings[i] = Mapping(**m)
        return mappings


class AmbiguousMappingSet(BaseMappingSet):
    """Mapping set of ambiguous IDs or labels.

    An entry is ambiguous when the same string appears both as a current
    primary identifier/label (in the datasource's full primary set) and
    as a secondary identifier/label in the mapping set.

    Attributes:
        ambiguous_ids: Set of ID strings that are ambiguous.
        ambiguous_labels: Set of label strings that are ambiguous.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise with empty ambiguous-ID/label stores."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "ambiguous_ids", set())
        object.__setattr__(self, "ambiguous_labels", set())

    @property
    def _ambiguous_ids(self) -> Any:  # Fix type
        return object.__getattribute__(self, "ambiguous_ids")

    @property
    def _ambiguous_labels(self) -> Any:  # Fix type
        return object.__getattribute__(self, "ambiguous_labels")

    def save(
        self,
        fmt: str,
        output_path: Path | str | None = None,
        **kwargs: object,
    ) -> Path:
        """Write to any supported format by name (sssom/rdf/json/owl)."""
        shared = self._save_shared(fmt, output_path, **kwargs)
        if shared is not None:
            return shared
        raise ValueError(f"Unknown format {fmt!r}. Choose from: json, owl, rdf, sssom")


_AMBIGUOUS_PREFIX = "Ambiguous:"
_ORIGINAL_MARKER = " Original comment: "


def _mapping_conflicts(
    m: Mapping,
    mode: str,
    primary_ids: set[str],
    primary_labels: dict[str, set[str]],
) -> tuple[list[str], set[str], set[str]]:
    """Return ``(conflicts, ambiguous_ids, ambiguous_labels)`` for one mapping.

    In ``"id"`` mode a conflict arises when the mapping's ``subject_id`` is
    itself a current primary ID. In ``"label"`` mode a conflict arises when the
    ``subject_label`` is a current primary label of some object other than this
    mapping's own ``object_id``.
    """
    subj_id = str(getattr(m, "subject_id", None) or "")
    subj_label = str(getattr(m, "subject_label", None) or "")
    obj_id = str(getattr(m, "object_id", None) or "")
    conflicts: list[str] = []
    amb_ids: set[str] = set()
    amb_labels: set[str] = set()

    if mode == "id":
        if subj_id and subj_id in primary_ids:
            amb_ids.add(subj_id)
            conflicts.append(
                f"secondary '{subj_id}' is also a current primary ID"
                + (f" (this mapping resolves to '{obj_id}')" if obj_id else "")
            )
    else:
        ids_for_label = primary_labels.get(subj_label) if subj_label else None
        conflicting_ids = (ids_for_label - {obj_id}) if ids_for_label else set()
        if conflicting_ids:
            amb_labels.add(subj_label)
            conflict_list = ", ".join(sorted(conflicting_ids))
            conflicts.append(
                f"subject_label '{subj_label}' is also the primary label of {conflict_list}"
                + (f" (this mapping resolves to '{obj_id}')" if obj_id else "")
            )

    return conflicts, amb_ids, amb_labels


def _with_conflict_comment(m: Mapping, conflicts: list[str]) -> Mapping:
    """Return a copy of *m* with ambiguous ``comment`` records."""
    existing = str(getattr(m, "comment", None) or "")
    if existing.startswith(_AMBIGUOUS_PREFIX):
        original = existing.split(_ORIGINAL_MARKER, 1)[1] if _ORIGINAL_MARKER in existing else ""
    else:
        original = existing

    detail = "; ".join(conflicts)
    comment = f"{_AMBIGUOUS_PREFIX} {detail}." + (
        f"{_ORIGINAL_MARKER}{original}" if original else ""
    )

    m_fields = {
        k: getattr(m, k, None)
        for k in (f.name for f in dataclass_fields(m))
        if getattr(m, k, None) is not None
    }
    m_fields["comment"] = comment
    return Mapping(**m_fields)


def _resolve_primary_sets(
    mappings: list[Mapping],
    primary_ids: set[str] | None,
    primary_labels: dict[str, set[str]] | None,
    mode: str,
) -> tuple[set[str], dict[str, set[str]]]:
    """Return the ``(primary_ids, primary_labels)`` to check *mappings* against.

    Explicit sets are used when supplied; otherwise they are derived from the
    ``object_id``/``object_label`` fields of the mappings themselves.
    """
    if mode == "id":
        if primary_ids:
            return primary_ids, {}
        ids = {str(getattr(m, "object_id", None) or "") for m in mappings} - {""}
        return ids, {}

    if primary_labels:
        return set(), primary_labels
    labels: dict[str, set[str]] = {}
    for m in mappings:
        lbl = str(getattr(m, "object_label", None) or "")
        oid = str(getattr(m, "object_id", None) or "")
        if lbl and oid:
            labels.setdefault(lbl, set()).add(oid)
    return set(), labels


def _annotate_ambiguous_mappings(
    mappings: list[Mapping],
    primary_labels: dict[str, set[str]] | None = None,
    primary_ids: set[str] | None = None,
    mapping_type: str = "id",
) -> list[Mapping]:
    """Return a new list where ambiguous mappings carry an explanatory comment.

    For **id** mappings a mapping is ambiguous when its ``subject_id`` is also a
    current primary ID. For **label** mappings a mapping is ambiguous when its
    ``subject_label`` is the primary label of a different object.

    This function is called automatically by
    :meth:`BaseParser.create_mapping_set` so that every output format
    (SSSOM, RDF, JSON, OWL, …) includes the annotation without the caller
    having to invoke :func:`_find_ambiguous` explicitly.

    Args:
        mappings: The raw list of :class:`~sssom_schema.Mapping` objects.
        primary_labels: Full label-to-primary-IDs index, else derived from
            *mappings*.
        primary_ids: Full primary-ID set, else derived from *mappings*.
        mapping_type: ``"id"`` or ``"label"``; controls which fields are
            examined for ambiguity.

    Returns:
        A new list where ambiguous entries carry an ambiguity ``comment``;
        non-ambiguous entries are returned unchanged (same object).
    """
    if not mappings:
        return mappings

    ids, labels = _resolve_primary_sets(mappings, primary_ids, primary_labels, mapping_type)
    result: list[Mapping] = []
    for m in mappings:
        conflicts, _, _ = _mapping_conflicts(m, mapping_type, ids, labels)
        result.append(_with_conflict_comment(m, conflicts) if conflicts else m)
    return result


def _get_primary_sets(
    mapping_set: MappingSet, mappings: list[Mapping]
) -> tuple[set[str], dict[str, set[str]]]:
    """Return ``(primary_ids, label_index)`` for the given mapping set.

    ``label_index`` maps each label text to the set of primary IDs that carry
    that label.
    """
    stored_ids: set[str] | None = (
        getattr(mapping_set, "_primary_ids", None) or None
    )  # treat empty set as missing
    stored_labels: dict[str, set[str]] | None = (
        getattr(mapping_set, "_primary_labels", None) or None
    )  # treat empty dict as missing

    if stored_ids is None:
        stored_ids = {str(getattr(m, "object_id", None) or "") for m in mappings}
        stored_ids.discard("")
    if stored_labels is None:
        # Build label-index fallback from the mappings themselves.
        label_index: dict[str, set[str]] = {}
        for m in mappings:
            oid = str(getattr(m, "object_id", None) or "")
            lbl = str(getattr(m, "object_label", None) or "")
            if oid and lbl:
                label_index.setdefault(lbl, set()).add(oid)
        stored_labels = label_index

    return stored_ids, stored_labels


def _find_ambiguous(mapping_set: BaseMappingSet) -> AmbiguousMappingSet:
    """Identify mappings whose subject is also a current primary entry.

    Args:
        mapping_set: Any :class:`BaseMappingSet` (id- or label-based).

    Returns:
        :class:`AmbiguousMappingSet` whose mappings each carry a ``comment``
        explaining which primary terms the subject conflicts with.  The sets
        ``ambiguous_ids`` and ``ambiguous_labels`` are populated accordingly.
        Returns an empty :class:`AmbiguousMappingSet` when no ambiguities are
        found.
    """
    mode = mapping_set._ambiguity_mode
    mappings = list(mapping_set.mappings or [])
    primary_ids, primary_labels = _get_primary_sets(mapping_set, mappings)

    ambiguous_mappings = []
    ambiguous_ids: set[str] = set()
    ambiguous_labels: set[str] = set()
    replacements: dict[int, Mapping] = {}

    for m in mappings:
        conflicts, ids, labels = _mapping_conflicts(m, mode, primary_ids, primary_labels)
        ambiguous_ids |= ids
        ambiguous_labels |= labels
        if not conflicts:
            continue
        annotated = _with_conflict_comment(m, conflicts)
        ambiguous_mappings.append(annotated)
        replacements[id(m)] = annotated

    if replacements:
        mapping_set.mappings = [replacements.get(id(m), m) for m in mappings]

    kwargs = {}
    for attr in (
        "curie_map",
        "mapping_set_id",
        "mapping_set_title",
        "mapping_set_description",
        "license",
        "creator_id",
        "creator_label",
        "mapping_provider",
        "mapping_tool",
        "mapping_tool_version",
        "mapping_date",
        "subject_source",
        "subject_source_version",
        "object_source",
        "object_source_version",
    ):
        val = getattr(mapping_set, attr, None)
        if val is not None:
            kwargs[attr] = val

    result = AmbiguousMappingSet(mappings=ambiguous_mappings, **kwargs)

    object.__setattr__(result, "ambiguous_ids", ambiguous_ids)
    object.__setattr__(result, "ambiguous_labels", ambiguous_labels)

    result._compute_cardinalities()
    return result


class BaseParser(ABC):
    """Abstract base class for all datasource parsers.

    Each parser is responsible for reading files from a specific datasource
    and extracting a :class:`BaseMappingSet` of cross-references between two
    identifier/label spaces.
    """

    # To be overridden by subclasses
    datasource_name: str = ""
    default_source_url: str = ""
    #: Importable package holding this datasource family's ``*.yaml``
    #: config files. Set by the concrete framework subclass.
    config_package: ClassVar[str] = ""
    #: Which :class:`BaseMappingSet` subclass :meth:`create_mapping_set`
    #: instantiates for each ``mapping_type``. The concrete framework
    #: subclass overrides this with its own mapping-set classes.
    mapping_set_classes: ClassVar[dict[str, type[BaseMappingSet]]] = {
        "id": BaseMappingSet,
        "label": BaseMappingSet,
    }
    #: Recorded as the SSSOM ``mapping_tool_version`` of generated mapping
    #: sets. Set by the concrete framework subclass to its own version.
    mapping_tool_version: ClassVar[str] = ""
    _config: DatasourceConfig | None = None

    def __init__(
        self,
        version: str | None = None,
        show_progress: bool = True,
        config_name: str | None = None,
    ):
        """Initialize the parser.

        Args:
            version: Version/release identifier for the datasource.
            show_progress: Whether to show progress bars during parsing.
            config_name: Name of config file to load (defaults to class name).
        """
        self.version = version
        self.show_progress = show_progress
        # Release date of the source data, used for the SSSOM ``mapping_date``.
        # Set by the download layer to the upstream release date; falls back
        # to the version when that is an ISO date (e.g. quarterly
        # date-versioned archives) or to today as a last resort.
        self.release_date: str | date | datetime | None = None

        # Load config from YAML if available
        if config_name:
            self._config = get_datasource_config(config_name, config_package=self.config_package)
        elif self.datasource_name:
            try:
                self._config = get_datasource_config(
                    self.datasource_name.lower(), config_package=self.config_package
                )
            except FileNotFoundError:
                self._config = None

    @property
    def config(self) -> DatasourceConfig | None:
        """Get the loaded configuration."""
        return self._config

    def get_download_url(self, key: str) -> str | None:
        """Get a download URL from config by key."""
        if self._config:
            return self._config.download_urls.get(key)
        return None

    def get_curie_map(self) -> dict[str, str]:
        """Get the CURIE map from config."""
        if self._config and self._config.mappingset_metadata:
            result: dict[str, str] = self._config.mappingset_metadata.get("curie_map", {})
            return result
        return {}

    def get_mappingset_metadata(self) -> dict[str, Any]:
        """Get mapping set metadata from config."""
        if self._config:
            result: dict[str, Any] = self._config.mappingset_metadata
            return result
        return {}

    def get_mapping_metadata(self) -> dict[str, Any]:
        """Get mapping metadata from config."""
        if self._config:
            result: dict[str, Any] = self._config.mapping_metadata
            return result
        return {}

    def load_metadata(self, yaml_path: str) -> dict[str, Any]:
        """Load metadata from a YAML config file."""
        with open(yaml_path, encoding="utf-8") as f:
            result: dict[str, Any] = yaml.safe_load(f)
            return result

    def apply_metadata_to_mappingset(
        self,
        mappingset: MappingSet,
        metadata: dict[str, Any],
    ) -> None:
        """Apply metadata to a MappingSet and its Mappings."""
        # Set MappingSet fields
        for key, value in metadata.get("mappingset", {}).items():
            if hasattr(mappingset, key) and value is not None:
                setattr(mappingset, key, value)
        # Set Mapping fields
        if hasattr(mappingset, "mappings") and mappingset.mappings:
            for mapping in mappingset.mappings:
                for key, value in metadata.get("mapping", {}).items():
                    if hasattr(mapping, key) and value is not None:
                        setattr(mapping, key, value)

    @staticmethod
    def _pair_hash(pri: str, sec: str) -> str:
        """See :func:`pair_hash`."""
        return pair_hash(pri, sec)

    def _record_id(self, namespace: str, pri: str, sec: str) -> str:
        """See :func:`mint_record_id`."""
        return mint_record_id(pri, sec, namespace=namespace)

    def _product_slug(self) -> str | None:
        """Extra IRI path segment identifying the run's data product.

        ``None`` for most parsers (one release == one product). Override
        when a parser option selects a different dataset rather
        than just a different output mode, e.g. a species selector for a
        multi-species datasource, where the same release number produces a
        disjoint set of mappings per species. Folded into ``mapping_set_id``
        and :meth:`_record_namespace` so two runs that differ only in this
        option don't collide on either IRI.
        """
        return None

    def _record_namespace(self) -> str:
        """Return this run's ``record_id`` namespace: ``{base}/{version}/{slug}/``.

        Mirrors ``mapping_set_id``'s ``{base}/{version}/{slug}`` ordering
        (see :meth:`create_mapping_set`) so a mapping's ``record_id`` is
        scoped to the same release/product as the mapping *set* it's
        asserted in. Use this, instead of reading
        ``mapping_metadata()["record_id"]`` directly, when building
        per-row ``record_id`` values.
        """
        base = str(self.get_mapping_metadata().get("record_id") or "")
        version = str(self.version) if self.version else None
        parts = [p for p in (version, self._product_slug()) if p]
        return base + "".join(f"{p}/" for p in parts)

    def _extract_version_from_file(self, file_path: Path) -> str | None:
        """Extract a version string embedded in a data file's header.

        Override in subclasses where the source file contains release
        metadata (e.g. ``Release: 2026_01`` in UniProt flat files).

        Args:
            file_path: Path to the data file to inspect.

        Returns:
            Version string, or ``None`` if not found.
        """
        return None

    def _resolve_version(self, file_path: Path | None = None) -> str:
        """Resolve the dataset version to use for source version fields.

        Resolution order:
        1. ``self.version`` if already set explicitly.
        2. Version extracted from file header via ``_extract_version_from_file``.
        3. ISO date or release token found in the filename stem
           (e.g. ``withdrawn_2026-04-07.txt`` -> ``2026-04-07``,
           ``chebi_245.sdf`` -> ``245``).
        4. File modification date (ISO-8601) when a path is provided.
        5. Today's date as a last resort.

        Sets ``self.version`` to the resolved value so that
        ``create_mapping_set`` picks it up for ``subject_source_version`` /
        ``object_source_version`` automatically.

        Args:
            file_path: Optional path to the primary input file.

        Returns:
            Resolved version string.
        """
        if self.version:
            return self.version

        if file_path is not None:
            file_path = Path(file_path)
            # 1. Try header-embedded version (parser-specific override)
            extracted = self._extract_version_from_file(file_path)
            if extracted:
                self.version = extracted
                return self.version
            # 2. Try ISO date (YYYY-MM-DD) in the filename stem
            iso_match = re.search(r"\d{4}-\d{2}-\d{2}", file_path.stem)
            if iso_match:
                self.version = iso_match.group(0)
                return self.version
            # 3. Try a plain numeric/semver token in the filename stem
            #    e.g. "chebi_245" -> "245", "gene_history_v2" -> "2"
            num_match = re.search(r"(?<![.\d])(\d{3,})(?![.\d])", file_path.stem)
            if num_match:
                self.version = num_match.group(1)
                return self.version
            # 4. Fall back to file modification time
            try:
                mtime = file_path.stat().st_mtime
                self.version = date.fromtimestamp(mtime).isoformat()
                return self.version
            except OSError:
                pass

        self.version = date.today().isoformat()
        return self.version

    @abstractmethod
    def parse(self, input_path: Path | str | None) -> MappingSet:
        """Parse the input file(s) and return a MappingSet.

        Args:
            input_path: Path to the input file or directory.

        Returns:
            A MappingSet containing all extracted mappings.
        """

    def _progress(
        self,
        iterable: Iterable[_T],
        desc: str | None = None,
        total: int | None = None,
    ) -> Iterable[_T]:
        """Wrap an iterable with a progress bar if enabled.

        Args:
            iterable: The iterable to wrap.
            desc: Description for the progress bar.
            total: Total number of items (if known).

        Returns:
            The iterable, optionally wrapped in tqdm.
        """
        if self.show_progress:
            return cast(Iterable[_T], tqdm(iterable, desc=desc, total=total))
        return iterable

    def _label_predicate_for_type(self, label_type: str) -> dict[str, str]:
        """Return predicate fields for a label mapping type.

        Used by :meth:`_build_mappings` when a row carries a ``_label_type``
        key.

        - ``"previous"``: the secondary name, label or label ``IAO:0100001`` "term replaced by").
        - ``"alias"`` (or any other value): a valid alternative name: ``oboInOwl:hasExactSynonym``.

        Args:
            label_type: ``"previous"`` or ``"alias"``.

        Returns:
            Dict with at least ``predicate_id`` and, where available,
            ``predicate_label``.
        """
        if label_type == "previous":
            m_meta = self.get_mapping_metadata()
            result: dict[str, str] = {"predicate_id": m_meta["predicate_id"]}
            pred_label = m_meta.get("predicate_label")
            if pred_label:
                result["predicate_label"] = str(pred_label)
            return result
        return {
            "predicate_id": "oboInOwl:hasExactSynonym",
            "predicate_label": "has exact synonym",
        }

    def _finalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Resolve ``_label_type`` into predicate fields and remove the key.

        If the row contains a ``_label_type`` entry and does not already
        have an explicit ``predicate_id``, the appropriate predicate fields
        are injected via :meth:`_label_predicate_for_type`.  The sentinel key
        is always removed before the row is used to construct a
        :class:`~sssom_schema.Mapping`.

        Args:
            row: Merged row dict (may contain ``_label_type``).

        Returns:
            The same dict, mutated in-place and returned for convenience.
        """
        label_type = row.pop("_label_type", None)
        if label_type is not None and "predicate_id" not in row:
            row.update(self._label_predicate_for_type(label_type))
        return row

    def _read_tsv(self, file_path: Path) -> pl.DataFrame:
        """Read a tab-separated flat file into a Polars DataFrame.

        Args:
            file_path: Path to the TSV file.

        Returns:
            The parsed :class:`polars.DataFrame`.
        """
        import polars as pl

        return pl.read_csv(
            file_path,
            separator="\t",
            infer_schema_length=10000,
            null_values=[""],
        )

    def _fixed_mapping_fields(self) -> dict[str, Any]:
        """Return the provenance fields shared by every mapping of one parse.

        Selects the mapping-level fields from the parser's config metadata
        for use as ``fixed_fields`` in :meth:`_build_mappings`.
        """
        m_meta = self.get_mapping_metadata()
        return {
            "mapping_justification": m_meta["mapping_justification"],
            "subject_source": m_meta.get("subject_source"),
            "object_source": m_meta.get("object_source"),
            "mapping_tool": m_meta.get("mapping_tool"),
            "license": m_meta.get("license"),
        }

    def _build_mappings(
        self,
        rows: Iterable[dict[str, Any]],
        fixed_fields: dict[str, Any] | None = None,
        *,
        desc: str = "Building mappings",
        total: int | None = None,
    ) -> list[Mapping]:
        """Build SSSOM Mapping objects from row dicts.

        Automatically injects mapping-level fields from the parser's
        config metadata (e.g. ``confidence``) unless the caller already
        provides them in ``fixed_fields`` or individual row dicts.

        Rows may carry a special ``_label_type`` key (``"alias"`` or
        ``"previous"``) instead of an explicit ``predicate_id``; the base
        class will resolve it to the correct predicate via
        :meth:`_label_predicate_for_type` before constructing the
        :class:`~sssom_schema.Mapping`.

        Args:
            rows: Per-row fields as dicts (subject_id, object_id, etc.).
            fixed_fields: Fields shared by all rows (predicate_id, license, etc.).
            desc: Progress bar description.
            total: Total count for the progress bar.

        Returns:
            List of Mapping objects.
        """
        _auto_fields = ("confidence",)
        m_meta = self.get_mapping_metadata()
        auto: dict[str, Any] = {
            k: m_meta[k] for k in _auto_fields if k in m_meta and m_meta[k] is not None
        }

        # Build base
        base: dict[str, Any] = {**auto, **(fixed_fields or {})}

        if base:
            merged: Iterable[dict[str, Any]] = (self._finalize_row({**base, **row}) for row in rows)
        else:
            merged = (self._finalize_row(dict(row)) for row in rows)
        return [Mapping(**row) for row in self._progress(merged, desc=desc, total=total)]

    def _build_comment(
        self,
        base_comment: str,
        additional: str | None = None,
    ) -> str:
        """Build a comment string with version information.

        Args:
            base_comment: The base comment text.
            additional: Additional text to append.

        Returns:
            The complete comment string.
        """
        parts = [base_comment] if base_comment else []
        if additional:
            parts.append(additional)
        if self.version:
            parts.append(f"Release: {self.version}.")
        return " ".join(parts)

    def _find_merged_column(
        self,
        columns: list[str],
        merged_info_patterns: list[str],
    ) -> str | None:
        """Find the merged info column regardless of naming variant."""
        normalized_patterns = [p.lower() for p in merged_info_patterns]
        for col in columns:
            normalized = self._normalize_column_name(col)
            if normalized in normalized_patterns:
                return col
            # Also check for partial match on key identifying part
            if "merged_into_report" in normalized:
                return col
        return None

    @staticmethod
    def _normalize_column_name(col: str) -> str:
        """Normalize column name for case-insensitive matching."""
        return col.lower().strip()

    @staticmethod
    def _find_column(columns: list[str], name: str) -> str | None:
        """Find column by case-insensitive name."""
        lower_name = name.lower()
        for col in columns:
            if col.lower() == lower_name:
                return col
        return None

    @staticmethod
    def normalize_withdrawn_id(subject_id: str | None) -> str:
        """Normalize a primary ID, converting empty/null to withdrawn.

        Args:
            subject_id: The raw primary identifier from the source file.

        Returns:
            The normalized primary ID, or WITHDRAWN_ENTRY for empty values.
        """
        if not subject_id or subject_id in ("-", ""):
            return WITHDRAWN_ENTRY
        return subject_id

    @staticmethod
    def is_withdrawn(identifier: str) -> bool:
        """Return if is withdrawn."""
        return WITHDRAWN_ENTRY == identifier

    @staticmethod
    def _split_labels(labels_str: str, sep: str = "|") -> list[str]:
        """Split a separated string of labels."""
        if not labels_str:
            return []
        return [s.strip() for s in labels_str.split(sep) if s.strip()]

    @staticmethod
    def is_withdrawn_primary(id: str) -> bool:
        """Check if an ID represents a withdrawn/deleted entry.

        Args:
            id: The primary identifier to check.

        Returns:
            True if the primary ID indicates a withdrawn entry.
        """
        return id == WITHDRAWN_ENTRY

    @staticmethod
    def _parse_merged_info(merged_str: str) -> tuple[str, str] | None:
        """Parse merged_into_report to extract hgnc_id and label.

        Returns (hgnc_id, label) or None if parsing fails.
        """
        if not merged_str or merged_str == "":
            return None
        # Try pipe separator first then slash
        if "|" in merged_str:
            parts = merged_str.split("|")
        else:
            parts = merged_str.split("/")
        if len(parts) >= 2:
            return (parts[0].strip(), parts[1].strip())
        return None

    def _resolve_mapping_date(self) -> str:
        """Resolve the SSSOM ``mapping_date`` for the output mapping set.

        The mapping date reflects when the source data was released, not when
        the mapping set was generated. Resolution order:

        1. ``self.release_date`` when set by the download layer (the upstream
           release date, e.g. an HTTP ``Last-Modified`` or archive date).
        2. ``self.version`` when it is an ISO date (``YYYY-MM-DD``), which is
           the most specific signal for sources whose version *is* a date
           (e.g. a quarterly date-versioned archive).
        3. Today's date as a last resort (e.g. live SPARQL queries).

        Returns:
            ISO-8601 date string (``YYYY-MM-DD``).
        """
        rd = self.release_date
        if isinstance(rd, datetime):
            return rd.date().isoformat()
        if isinstance(rd, date):
            return rd.isoformat()
        if isinstance(rd, str) and rd:
            return rd
        if self.version and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(self.version)):
            return str(self.version)
        return date.today().isoformat()

    def _source_version(self) -> str | None:
        """Return the value for the set-level SSSOM ``*_source_version`` fields.

        Always the analyzed release, :attr:`version`. The same field means
        the same thing across every datasource's mapping set.
        """
        return self.version

    def create_mapping_set(
        self,
        mappings: list[Mapping],
        mapping_type: str = "id",
        *,
        primary_ids: set[str] | None = None,
        primary_labels: dict[str, set[str]] | None = None,
    ) -> BaseMappingSet:
        """Create a mapping set instance with config metadata.

        Common factory method for creating mapping sets with
        all SSSOM metadata populated from the YAML config. It also computes
        cardinalities for mappings.

        Args:
            mappings: List of SSSOM Mapping objects.
            mapping_type: "id" for cardinality by ID, "label" for
                cardinality by label; also selects the mapping-set class
                via :attr:`mapping_set_classes`.
            primary_ids: Full set of current primary IDs, when the parser has
                access to the complete-set file. Stored on the result to drive
                ``to_pri_ids``/ambiguity checks.
            primary_labels: Full label->primary-IDs map, the label-side
                counterpart to primary_ids.

        Returns:
            MappingSet with computed cardinalities.
        """
        import curies as _curies

        ms_meta = self.get_mappingset_metadata()
        curie_map = self.get_curie_map()

        # Build a converter from the curie_map
        converter = _curies.Converter.from_prefix_map(curie_map)

        def _compress(val: Any) -> Any:
            """Compress a URI string or list of URI strings to CURIEs."""
            if isinstance(val, str):
                return converter.compress(val) or val
            if isinstance(val, list):
                return [converter.compress(v) or v if isinstance(v, str) else v for v in val]
            return val

        mapping_set_class = self.mapping_set_classes.get(
            mapping_type, self.mapping_set_classes["id"]
        )

        # Build description with version if available
        description = ms_meta.get("mapping_set_description", "")
        if self.version and description:
            description = f"{description} Version: {self.version}."

        # Annotate ambiguous mappings (primary also appears as secondary)
        mappings = _annotate_ambiguous_mappings(mappings, mapping_type=mapping_type)
        # Source-version carries the genome build, not the release (the
        # release is already explicit in mapping_set_version/_id).
        source_version = self._source_version()
        product_slug = self._product_slug()
        version_path = f"/{self.version}/{product_slug}" if product_slug else f"/{self.version}"
        fix_ms_id = str(ms_meta.get("mapping_set_id")) + version_path
        # Create the mapping set with SSSOM metadata
        mapping_set = mapping_set_class(
            mappings=mappings,
            curie_map=curie_map,
            mapping_set_id=fix_ms_id,
            mapping_set_version=self.version,
            mapping_set_title=ms_meta.get("mapping_set_title"),
            mapping_set_description=description or None,
            creator_id=_compress(ms_meta.get("creator_id")),
            creator_label=ms_meta.get("creator_label"),
            comment=ms_meta.get("comment"),
            license=_compress(ms_meta.get("license")),
            subject_source=ms_meta.get("subject_source"),
            subject_source_version=source_version,
            object_source=ms_meta.get("object_source"),
            object_source_version=source_version,
            mapping_provider=_compress(ms_meta.get("mapping_provider")),
            mapping_tool=_compress(ms_meta.get("mapping_tool")),
            mapping_tool_version=self.mapping_tool_version or None,
            mapping_date=self._resolve_mapping_date(),
            see_also=_compress(ms_meta.get("see_also")),
            issue_tracker=_compress(ms_meta.get("issue_tracker")),
            subject_preprocessing=_compress(ms_meta.get("subject_preprocessing")),
            object_preprocessing=_compress(ms_meta.get("object_preprocessing")),
        )
        # Annotate ambiguous mappings (primary also appears as secondary)
        if mapping_type == "label":
            pri_labels = mapping_set._primary_labels
            mappings_updated = _annotate_ambiguous_mappings(
                mappings, mapping_type=mapping_type, primary_labels=pri_labels
            )
        else:
            pri_ids = mapping_set._primary_ids
            mappings_updated = _annotate_ambiguous_mappings(
                mappings, mapping_type=mapping_type, primary_ids=pri_ids
            )
        mapping_set.mappings = mappings_updated
        # Compute cardinalities
        mapping_set._compute_cardinalities(on=mapping_type)

        if primary_ids is not None:
            object.__setattr__(mapping_set, "_primary_ids", primary_ids)
        if primary_labels is not None:
            object.__setattr__(mapping_set, "_primary_labels", primary_labels)

        return mapping_set

    def load(self, path: Path | str, *, mapping_type: str | None = None) -> BaseMappingSet:
        """Load an SSSOM file back into this parser's own mapping-set class.

        Returns the same object :meth:`create_mapping_set` produces. The class
        is taken from *mapping_type*, else inferred as ``"id"`` when the
        mappings carry a ``subject_id`` and ``"label"`` when they don't. A
        ``mapping_set_id`` that doesn't match this datasource's configured base
        is warned about but still loaded.

        Args:
            path: Path to the SSSOM TSV file.
            mapping_type: ``"id"``/``"label"`` to force the class, or ``None``
                to infer it from the mappings.

        Returns:
            The datasource's :attr:`mapping_set_classes` instance for the
            resolved type, with cardinalities computed.
        """
        from sssom.parsers import parse_sssom_table, to_mapping_set_document

        from mapkgsutils.exports import _mapping_set_from_document

        msdf = parse_sssom_table(str(path))
        src = to_mapping_set_document(msdf).mapping_set

        found = str(getattr(src, "mapping_set_id", "") or "")
        expected = str(self.get_mappingset_metadata().get("mapping_set_id") or "")
        if expected and not found.startswith(expected):
            logger.warning(
                "Loaded mapping_set_id %r does not match %s base %r",
                found,
                self.datasource_name,
                expected,
            )

        if mapping_type is None:
            has_subject_id = any(getattr(m, "subject_id", None) for m in (src.mappings or []))
            mapping_type = "id" if has_subject_id else "label"
        mapping_set_class = self.mapping_set_classes.get(
            mapping_type, self.mapping_set_classes["id"]
        )

        mapping_set = _mapping_set_from_document(src, msdf.converter, mapping_set_class)
        mapping_set._compute_cardinalities(on=mapping_type)
        return mapping_set


class ProductSlugMixin:
    """Mixin for datasources whose release splits into disjoint data products.

    Set :attr:`product_attr` to the instance attribute holding the run's
    selector; its value becomes the :meth:`BaseParser._product_slug` folded
    into ``mapping_set_id``/``record_id``, so two runs differing only in that
    option don't collide on either IRI. Mix in ahead of the parser base.

    Example:
        >>> class SpeciesAwareMixin(ProductSlugMixin):
        ...     product_attr = "species"
    """

    #: Instance attribute selecting the data product (e.g. ``"species"``).
    product_attr: ClassVar[str] = ""

    def _product_slug(self) -> str | None:
        """Value of :attr:`product_attr` on this run, or ``None`` if unset."""
        if not self.product_attr:
            return None
        return cast("str | None", getattr(self, self.product_attr, None))


__all__ = [
    "WITHDRAWN_ENTRY",
    "WITHDRAWN_ENTRY_LABEL",
    "AmbiguousMappingSet",
    "BaseDownloader",
    "BaseMappingSet",
    "BaseParser",
    "DatasourceConfig",
    "DistributionEra",
    "ProductSlugMixin",
    "XrefSource",
    "get_datasource_config",
    "load_config",
    "mint_record_id",
    "pair_hash",
]
