"""Base parser/downloader framework and mapping-set classes."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import date, datetime
from importlib import resources as _importlib_resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast

import yaml
from sssom_schema import Mapping, MappingCardinalityEnum, MappingSet
from tqdm import tqdm

from mapkgsutils.logging import logger

if TYPE_CHECKING:
    import rdflib
    from sssom import sssom_document

_T = TypeVar("_T")

# Values for withdrawn entries

WITHDRAWN_ENTRY = "sssom:NoTermFound"
WITHDRAWN_ENTRY_LABEL = "Withdrawn Entry"


@dataclass
class DistributionEra:
    """One historical "shape" a datasource's distribution has taken.

    Lets a config describe multiple eras (different URL templates, formats,
    or archive locations) instead of a single hardcoded threshold. Eras are
    matched by version using from_version/to_version (inclusive, numeric-aware
    comparison so "100" < "245" compares correctly; falls back to lexicographic
    for date-string versions like HGNC's "YYYY-MM-DD").
    """

    id: str
    download_urls: dict[str, str] = field(default_factory=dict)
    archive_url: str = ""
    format: str | None = None
    from_version: str | None = None
    to_version: str | None = None
    wayback: bool = False  # declarative only for now -- no resolver yet


@dataclass
class XrefSource:
    """A suggested cross-reference crosswalk source for a datasource.

    Passed to :func:`mapkgsutils.context.load_xref_mapping` after downloading
    *url* and renaming *object_id_col*/*object_label_col*/the chosen
    *subject_id_cols* entry to ``object_id``/``object_label``/``subject_id``.
    """

    id: str
    name: str = ""
    url: str = ""
    format: str = "tsv"
    object_id_col: str = "object_id"
    object_label_col: str = "object_label"
    subject_id_cols: dict[str, str] = field(default_factory=dict)
    note: str = ""


def _cmp_versions(a: str, b: str) -> int:
    """Compare two version strings, numerically when possible.

    Falls back to plain string comparison for non-numeric versions (e.g.
    ISO date strings like ``"2026-04-07"``, which already sort correctly
    lexicographically).

    Returns:
        Negative if ``a < b``, zero if equal, positive if ``a > b``.
    """
    try:
        ai, bi = int(a), int(b)
        return (ai > bi) - (ai < bi)
    except ValueError:
        return (a > b) - (a < b)


@dataclass
class DatasourceConfig:
    """Configuration for a biological database datasource loaded from YAML."""

    name: str
    prefix: str
    curie_base_url: str
    # Proper schema fields
    config_id: str = ""
    datasource_id: str = ""
    parser_class: str = ""
    parse_options: dict[str, Any] = field(default_factory=dict)
    mapping_sets: dict[str, Any] = field(default_factory=dict)
    # Old field, remove at some point
    available_outputs: list[str] = field(default_factory=list)
    default_output_filename: str = ""
    download_urls: dict[str, Any] = field(default_factory=dict)
    primary_file_key: str = ""
    id_pattern: str = ""
    archive_url: str = ""
    input_file_types: list[str] = field(default_factory=list)
    source: str = ""
    homepage: str = ""
    data_license: str = ""
    # SPARQL-based datasources (e.g., Wikidata)
    sparql_endpoint: str = ""
    queries: dict[str, str] = field(default_factory=dict)
    # For now, only ChEBI: version threshold for new TSV format.
    # Use if release files change location or serialization.
    new_format_version: int | None = None
    # Historical distribution "shapes" this datasource has had (different URL
    # templates, formats, or archive locations across its lifetime).
    distribution_eras: list[DistributionEra] = field(default_factory=list)
    # Suggested cross-reference crosswalk sources
    xref_sources: list[XrefSource] = field(default_factory=list)
    # Species this datasource publishes
    species: dict[str, Any] = field(default_factory=dict)
    # Genome assembly/build metadata.
    genome_build: dict[str, Any] = field(default_factory=dict)
    # Compound/entry subset this datasource publishes (e.g. ChEBI's
    # 3star/complete). Generic, config-driven counterpart to `species`.
    subset: dict[str, Any] = field(default_factory=dict)
    # Full metadata from YAML
    mappingset_metadata: dict[str, Any] = field(default_factory=dict)
    mapping_metadata: dict[str, Any] = field(default_factory=dict)

    def species_token(self, taxon_id: str | int) -> str:
        """Resolve a canonical NCBI taxon ID to this datasource's own species token.

        Reads the ``species.available`` block (see ``ensembl.yaml``), which
        maps each supported taxon ID to the datasource-specific token used to
        build download paths/filters (e.g. Ensembl's ``homo_sapiens``).

        Args:
            taxon_id: Canonical NCBI taxon ID, e.g. ``9606`` or ``"9606"``.

        Returns:
            The datasource-specific species token.

        Raises:
            ValueError: If no ``species`` block is configured, or *taxon_id*
                is not one of its declared entries.
        """
        available = {str(k): v for k, v in ((self.species or {}).get("available") or {}).items()}
        entry = available.get(str(taxon_id))
        if entry is None:
            known = ", ".join(sorted(available)) or "(none configured)"
            raise ValueError(
                f"Unknown species taxon ID {taxon_id!r} for {self.name!r}. Known: {known}"
            )
        return str(entry["token"])

    def default_species(self) -> str | int:
        """Return the configured default species taxon ID (``9606`` if unset)."""
        return cast("str | int", (self.species or {}).get("default", 9606))

    def default_subset(self) -> str | None:
        """Return the configured default subset, or ``None`` if this datasource has none."""
        return cast("str | None", (self.subset or {}).get("default"))

    def xref_source(self, source_id: str) -> XrefSource | None:
        """Return the configured :class:`XrefSource` with id *source_id*, if any."""
        for src in self.xref_sources:
            if src.id == source_id:
                return src
        return None

    def formats_for(self, kind: str) -> Any:
        """Return the list of supported output formats for a mapping-set kind.

        Args:
            kind: Mapping-set key, e.g. ``"ids"`` or ``"labels"``.

        Returns:
            List of format strings, or an empty list when the kind is absent.
        """
        return self.mapping_sets.get(kind, {}).get("formats", [])

    def era_for(self, version: str | None) -> DistributionEra | None:
        """Return the first configured era whose bounds contain *version*.

        Args:
            version: Version string to match, or ``None``.

        Returns:
            The matching :class:`DistributionEra`, or ``None`` if no eras are
            configured or none match (callers should fall back to the
            top-level ``download_urls``/``new_format_version`` behavior).
        """
        if not self.distribution_eras or version is None:
            return None
        for era in self.distribution_eras:
            if era.from_version is not None and _cmp_versions(version, era.from_version) < 0:
                continue
            if era.to_version is not None and _cmp_versions(version, era.to_version) > 0:
                continue
            return era
        return None


def load_config(datasource_name: str, *, config_package: str) -> dict[str, Any]:
    """Load configuration from a YAML file for a datasource.

    Args:
        datasource_name: Name of the datasource (e.g., 'chebi', 'hgnc').
        config_package: Importable package holding the datasource's
            ``*.yaml`` config files (e.g. ``"pysec2pri.config"``).

    Returns:
        Dictionary with the full YAML configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    from mapkgsutils.config.schema import validate_config_dict

    config_dir = Path(_importlib_resources.files(config_package))  # type: ignore[arg-type]
    config_path = config_dir / f"{datasource_name.lower()}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        result: dict[str, Any] = yaml.safe_load(f)
    validate_config_dict(result, config_path.name)
    return result


def get_datasource_config(datasource_name: str, *, config_package: str) -> DatasourceConfig:
    """Load and parse a DatasourceConfig from YAML.

    Args:
        datasource_name: Name of the datasource (e.g., 'chebi', 'hgnc').
        config_package: Importable package holding the datasource's
            ``*.yaml`` config files (e.g. ``"pysec2pri.config"``).

    Returns:
        DatasourceConfig object populated from YAML.
    """
    raw = load_config(datasource_name, config_package=config_package)

    eras = [
        DistributionEra(
            id=era.get("id", ""),
            download_urls=era.get("download_urls") or {},
            archive_url=era.get("archive_url", ""),
            format=era.get("format"),
            from_version=era.get("from_version"),
            to_version=era.get("to_version"),
            wayback=era.get("wayback", False),
        )
        for era in raw.get("distribution_eras", [])
    ]

    xref_sources = [
        XrefSource(
            id=src.get("id", ""),
            name=src.get("name", ""),
            url=src.get("url", ""),
            format=src.get("format", "tsv"),
            object_id_col=src.get("object_id_col", "object_id"),
            object_label_col=src.get("object_label_col", "object_label"),
            subject_id_cols=src.get("subject_id_cols") or {},
            note=src.get("note", ""),
        )
        for src in raw.get("xref_sources", [])
    ]

    return DatasourceConfig(
        name=raw.get("name", ""),
        prefix=raw.get("prefix", ""),
        curie_base_url=raw.get("curie_base_url", ""),
        config_id=raw.get("config_id", ""),
        datasource_id=raw.get("datasource_id", ""),
        parser_class=raw.get("parser_class", ""),
        parse_options=raw.get("parse_options") or {},
        mapping_sets=raw.get("mapping_sets") or {},
        available_outputs=raw.get("available_outputs", []),
        default_output_filename=raw.get("default_output_filename", ""),
        download_urls=raw.get("download_urls", {}),
        primary_file_key=raw.get("primary_file_key", ""),
        id_pattern=raw.get("id_pattern", ""),
        archive_url=raw.get("archive_url", ""),
        input_file_types=raw.get("input_file_types", []),
        source=raw.get("source", ""),
        homepage=raw.get("homepage", ""),
        data_license=raw.get("data_license", ""),
        sparql_endpoint=raw.get("sparql_endpoint", ""),
        queries=raw.get("queries", {}),
        new_format_version=raw.get("new_format_version"),
        distribution_eras=eras,
        xref_sources=xref_sources,
        species=raw.get("species") or {},
        genome_build=raw.get("genome_build") or {},
        subset=raw.get("subset") or {},
        mappingset_metadata=raw.get("mappingset", {}),
        mapping_metadata=raw.get("mapping", {}),
    )


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
        The base implementation raises :class:`ValueError` because most
        datasources only provide the latest release.

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
    or product (species/subset). This is the join key a cross-release
    consolidation layer uses to match a mapping across releases (to discover
    when it first/last appeared) -- *not* what ends up in the ``record_id``
    field (see :func:`mint_record_id`).
    """
    return hashlib.sha256(f"{pri}|{sec}".encode()).hexdigest()[:16]


def mint_record_id(pri: str, sec: str, *, namespace: str) -> str:
    """Mint a row's ``record_id`` -- the row's OWL Axiom IRI in SSSOM's RDF/OWL output.

    Scoped to *namespace* (typically a release- and product-specific prefix,
    see :meth:`BaseParser._record_namespace`), so the same (pri, sec) pair
    parsed from a different release/product gets a different ``record_id``.
    This matters because each :class:`~sssom_schema.Mapping` row is
    serialised as an ``owl:Axiom``: if record_id didn't vary across
    releases, loading several releases' SSSOM/RDF into one triplestore would
    assert contradictory axioms (different predicate, cardinality,
    confidence, ...) under the same IRI.

    The trailing 16 hex characters are always :func:`pair_hash`'s
    version-independent digest -- use that function directly (not this one)
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
        ms_id: str = str(getattr(self, "mapping_set_id", None) or "") + f"/{self.version}"
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
    as a secondary identifier/label in the mapping set (i.e. it is also
    recorded as a previous or alias term that *maps to something else*).

    Because the directionality of the mapping is unclear for such entries,
    the resolver leaves them blank and warns the user rather than silently
    overwriting data.

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


def _build_annotated_mapping(m: Mapping, new_comment: str) -> Mapping:
    """Return a copy of *m* with *new_comment* set as the ``comment`` field."""
    m_fields = {
        k: getattr(m, k, None)
        for k in (f.name for f in dataclass_fields(m))
        if getattr(m, k, None) is not None
    }
    m_fields["comment"] = new_comment
    return Mapping(**m_fields)


def _annotate_id_mappings(
    mappings: list[Mapping],
    primaries: set[str] | None = None,
) -> list[Mapping]:
    """Annotate ambiguous id mappings (subject_id is also an active primary ID)."""
    if not primaries:
        object_ids: set[str] = {str(getattr(m, "object_id", None) or "") for m in mappings} - {""}
    else:
        object_ids = primaries

    result: list[Mapping] = []
    for m in mappings:
        subj_id = str(getattr(m, "subject_id", None) or "")
        obj_id = str(getattr(m, "object_id", None) or "")
        if subj_id and subj_id in object_ids:
            existing = str(getattr(m, "comment", None) or "")
            if existing.startswith("Ambiguous mapping:"):
                result.append(m)
                continue
            new_comment = (
                f"Ambiguous mapping: secondary '{subj_id}' is also a current primary ID"
                + (f" (this mapping resolves to '{obj_id}')" if obj_id else "")
                + "."
                + (f" {existing}" if existing else "")
            )
            result.append(_build_annotated_mapping(m, new_comment))
        else:
            result.append(m)
    return result


def _annotate_label_mappings(
    mappings: list[Mapping],
    primary_labels: dict[str, set[str]] | None = None,
) -> list[Mapping]:
    """Annotate ambiguous label mappings."""
    if primary_labels:
        label_to_obj_ids = primary_labels
    else:
        label_to_obj_ids = {}
        for m in mappings:
            lbl = str(getattr(m, "object_label", None) or "")
            oid = str(getattr(m, "object_id", None) or "")
            if lbl and oid:
                label_to_obj_ids.setdefault(lbl, set()).add(oid)

    result: list[Mapping] = []
    for m in mappings:
        subj_label = str(getattr(m, "subject_label", None) or "")
        obj_id = str(getattr(m, "object_id", None) or "")
        ids_for_label = label_to_obj_ids.get(subj_label) if subj_label else None
        conflicting_ids = (ids_for_label - {obj_id}) if ids_for_label else set()
        if conflicting_ids:
            existing = str(getattr(m, "comment", None) or "")
            if existing.startswith("Ambiguous mapping:"):
                result.append(m)
                continue
            conflict_list = ", ".join(sorted(conflicting_ids))
            new_comment = (
                f"Ambiguous mapping: subject_label '{subj_label}' is also the"
                f" label of {conflict_list}"
                + (f" (this mapping resolves to '{obj_id}')" if obj_id else "")
                + "."
                + (f" Original comment: {existing}" if existing else "")
            )
            result.append(_build_annotated_mapping(m, new_comment))
        else:
            result.append(m)
    return result


def _annotate_ambiguous_mappings(
    mappings: list[Mapping],
    primary_labels: dict[str, set[str]] | None = None,
    primary_ids: set[str] | None = None,
    mapping_type: str = "id",
) -> list[Mapping]:
    """Return a new list where ambiguous mappings carry an explanatory comment.

    For **id** mappings a mapping is ambiguous when its ``subject_id`` also
    appears as an ``object_id`` in the list (the secondary ID is also a live
    primary ID).

    For **label** mappings a mapping is ambiguous when its ``subject_label``
    appears as a primary label for a *different* entity, determined by
    checking whether the same label text is paired with a different
    ``object_id`` elsewhere in the list.

    This function is called automatically by
    :meth:`BaseParser.create_mapping_set` so that every output format
    (SSSOM, RDF, JSON, OWL, …) includes the annotation without the caller
    having to invoke :func:`_find_ambiguous` explicitly.

    Args:
        mappings: The raw list of :class:`~sssom_schema.Mapping` objects.
        mapping_type: ``"id"`` or ``"label"``; controls which fields are
            examined for ambiguity.

    Returns:
        A new list where ambiguous entries have a ``comment`` prepended;
        non-ambiguous entries are returned unchanged (same object).
    """
    if not mappings:
        return mappings

    if mapping_type == "id":
        return _annotate_id_mappings(mappings, primary_ids)
    return _annotate_label_mappings(mappings, primary_labels)


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


def _extract_fields_ambig(m: Mapping) -> tuple[str, str, str, str, str, str]:
    return (
        str(getattr(m, "subject_id", None) or ""),
        str(getattr(m, "subject_label", None) or ""),
        str(getattr(m, "object_id", None) or ""),
        str(getattr(m, "object_label", None) or ""),
        str(getattr(m, "predicate_id", None) or ""),
        str(getattr(m, "comment", None) or ""),
    )


def _build_conflicts(
    subj_id: str | None,
    subj_label: str,
    obj_id: str,
    obj_label: str,
    pred: str,
    primary_ids: set[str],
    primary_labels: dict[str, set[str]],
    mode: str,
) -> tuple[list[str], set[str], set[str]]:
    """Return ``(conflicts, ambiguous_ids, ambiguous_labels)`` for one mapping.

    For **id** mode a conflict is raised when ``subj_id`` is present in
    ``primary_ids``.

    For labels a conflict is raised only when ``subj_label`` maps to a
    primary ID *other* than ``obj_id`` in ``primary_labels``.
    """
    conflicts = []
    amb_ids: set[str] = set()
    amb_labels: set[str] = set()
    if mode == "id":
        if subj_id and subj_id in primary_ids:
            amb_ids.add(subj_id)
            conflicts.append(
                f"secondary '{subj_id}' is also a current primary ID"
                + (f" (mapping points to '{obj_id}')" if obj_id else "")
            )
    if mode == "label":
        if subj_label in primary_labels:
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


def _make_annotated_mapping(m: Mapping, conflicts: list[str], existing_raw: str) -> Mapping:
    if existing_raw.startswith("Ambiguous mapping:"):
        # Already wrapped by _annotate_id_mappings/_annotate_label_mappings;
        # unwrap to the truly original comment to avoid nested wrapping.
        marker = " Original comment: "
        original = existing_raw.split(marker, 1)[1] if marker in existing_raw else ""
    else:
        original = existing_raw

    conflict_detail = "; ".join(conflicts)

    new_comment = f"Ambiguous: {conflict_detail}." + (
        f" Original comment: {original}" if original else ""
    )

    m_fields = {
        k: getattr(m, k, None)
        for k in (f.name for f in dataclass_fields(m))
        if getattr(m, k, None) is not None
    }
    m_fields["comment"] = new_comment
    return Mapping(**m_fields)


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
    get_subj = "subject_id" if mode == "id" else "object_id"
    mappings = list(mapping_set.mappings or [])
    primary_ids, primary_labels = _get_primary_sets(mapping_set, mappings)

    ambiguous_mappings = []
    ambiguous_ids = set()
    ambiguous_labels = set()

    for m in mappings:
        subj_id, subj_label, obj_id, obj_label, pred, raw_comment = _extract_fields_ambig(m)
        conflicts, ids, labels = _build_conflicts(
            subj_id, subj_label, obj_id, obj_label, pred, primary_ids, primary_labels, mode
        )

        ambiguous_ids |= ids
        ambiguous_labels |= labels
        if not conflicts:
            continue

        ambiguous_mappings.append(_make_annotated_mapping(m, conflicts, raw_comment))

    if ambiguous_mappings:
        annotated_by_subj = {
            str(getattr(am, "subject_id", None) or ""): am for am in ambiguous_mappings
        }
        annotated_by_label = {
            str(getattr(am, "subject_label", None) or ""): am for am in ambiguous_mappings
        }

        updated_source = []
        changed = False

        for m in mappings:
            subj_id = str(getattr(m, get_subj, None) or "")
            subj_label = str(getattr(m, "subject_label", None) or "")

            replacement = annotated_by_subj.get(subj_id) or annotated_by_label.get(subj_label)

            if replacement is not None and replacement is not m:
                updated_source.append(replacement)
                changed = True
            else:
                updated_source.append(m)

        if changed:
            mapping_set.mappings = updated_source

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
        # Genome assembly/build for the SSSOM ``*_source_version`` fields.
        # Default ``None`` -> resolved from config per species, or left to
        # fall back to ``self.version``. Parsers that read a build straight
        # from the data (e.g. an Ensembl-style ``mapping_session``) may set
        # this, or override per mapping row.
        self.genome_build: str | None = None
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
        when a parser option selects a genuinely different dataset rather
        than just a different output mode -- e.g. a species selector for a
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
        asserted in -- use this (instead of reading
        ``mapping_metadata()["record_id"]`` directly) when building
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

    def _species_build(self) -> str | None:
        """Return the configured genome build for this run's species, if any.

        Reads ``species.available[<species>].build`` from config.
        ``self.species`` may be a canonical taxon ID (single-species runs)
        or a datasource token (all-species runs), so both the ``available``
        key and each entry's ``token`` are matched.

        Returns:
            The per-species build string, or ``None`` when not configured.
        """
        cfg = self._config
        species = getattr(self, "species", None)
        if not cfg or species is None:
            return None
        available = (cfg.species or {}).get("available") or {}
        entry = available.get(species) or available.get(str(species))
        if entry is None:
            for value in available.values():
                if isinstance(value, dict) and str(value.get("token")) == str(species):
                    entry = value
                    break
        if isinstance(entry, dict) and entry.get("build"):
            return str(entry["build"])
        return None

    def _genome_build(self) -> str | None:
        """Resolve the genome assembly/build for the source-version fields.

        The mapping set's release is already explicit in
        ``mapping_set_version`` / ``mapping_set_id``, so
        ``subject_source_version`` / ``object_source_version`` carry the
        genome build (e.g. ``"GRCh38"``) rather than repeating the release.
        Resolution order:

        1. an explicit :attr:`genome_build` override (e.g. a build a parser
           discovered straight from the data);
        2. the per-species build from ``species.available[<species>].build``;
        3. the datasource-wide ``genome_build.default`` -- but *only* for
           single-build datasources (those with no ``species.available`` map,
           e.g. a human-only datasource). A multi-species datasource never
           falls through to ``default``, so it can't stamp one build (say
           GRCh38) onto a non-human species; an uncurated species is left to
           fall back to the release instead.

        Returns:
            The resolved build, or ``None`` when none is configured (e.g.
            ChEBI), so callers fall back to :attr:`version`.
        """
        if self.genome_build:
            return self.genome_build
        species_build = self._species_build()
        if species_build:
            return species_build
        cfg = self._config
        if not cfg or (cfg.species or {}).get("available"):
            return None
        default = (cfg.genome_build or {}).get("default")
        return str(default) if default else None

    def _source_version(self) -> str | None:
        """Return the value for the set-level SSSOM ``*_source_version`` fields.

        The genome build when one is resolvable (see :meth:`_genome_build`),
        otherwise the release :attr:`version` (unchanged behaviour for
        datasources without a genome build, e.g. ChEBI).
        """
        return self._genome_build() or self.version

    def create_mapping_set(
        self,
        mappings: list[Mapping],
        mapping_type: str = "id",
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

        return mapping_set


__all__ = [
    "WITHDRAWN_ENTRY",
    "WITHDRAWN_ENTRY_LABEL",
    "AmbiguousMappingSet",
    "BaseDownloader",
    "BaseMappingSet",
    "BaseParser",
    "DatasourceConfig",
    "DistributionEra",
    "XrefSource",
    "get_datasource_config",
    "load_config",
    "mint_record_id",
    "pair_hash",
]
