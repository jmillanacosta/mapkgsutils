"""Generic file-download primitives and release/version dispatch."""

from __future__ import annotations

import gzip
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import httpx
from tqdm import tqdm

from mapkgsutils.logging import logger

__all__ = [
    "CloudflareBlockedError",
    "ReleaseInfo",
    "check_release",
    "download_datasource",
    "download_datasource_with_release",
    "download_file",
    "download_urls",
    "get_download_urls",
    "get_file_last_modified",
    "get_latest_release_info",
    "iter_with_progress",
    "list_versions",
    "resolve_datasource_urls",
    "resolve_release_date",
]

_CLOUDFLARE_HINTS = (
    "cf-ray",
    "cloudflare",
    "cf-mitigated",
    "__cf_bm",
    "cf-request-id",
)


class CloudflareBlockedError(Exception):
    """Raised when a download is blocked by Cloudflare bot protection.

    Args:
        url: The URL that was blocked.
    """

    def __init__(self, url: str) -> None:
        """Store *url* and build a message explaining the manual-download workaround."""
        self.url = url
        super().__init__(
            f"Download of '{url}' was blocked by Cloudflare bot protection.\n"
            "Please download the file manually in a web browser and pass the "
            "local path as an argument instead.\n"
        )


def _is_cloudflare_blocked(response: httpx.Response) -> bool:
    """Return True if *response* looks like a Cloudflare block page."""
    if response.status_code in (403, 503):
        headers_lower = {k.lower() for k in response.headers}
        if any(hint in headers_lower for hint in _CLOUDFLARE_HINTS):
            return True
        # Cloudflare sometimes returns 403 with an HTML page even without the
        # header being present, check the body.
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            response.read()  # body is unread on a streamed response; .text needs it
            text = response.text
            if "cloudflare" in text.lower() or "cf-ray" in text.lower():
                return True
    return False


@dataclass
class ReleaseInfo:
    """Information about a datasource release."""

    datasource: str
    version: str | None
    release_date: datetime | None
    is_new: bool
    files: dict[str, str]  # key -> URL mapping


def download_file(
    url: str,
    output_path: Path,
    decompress_gz: bool = True,
    timeout: float | None = None,
    show_progress: bool = True,
    description: str | None = None,
) -> Path:
    """Download a file from URL to the specified path.

    Args:
        url: URL to download from.
        output_path: Where to save the file.
        decompress_gz: Whether to decompress .gz files automatically.
        timeout: Request timeout in seconds.
        show_progress: Whether to show a progress bar.
        description: Description for the progress bar.

    Returns:
        Path to the downloaded (and optionally decompressed) file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get filename for progress bar description
    if description is None:
        description = output_path.name
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            if _is_cloudflare_blocked(response):
                raise CloudflareBlockedError(url)
            response.raise_for_status()

            # Get total size if available
            total_size = int(response.headers.get("content-length", 0))

            # Determine if we need to decompress
            is_gzip = url.endswith(".gz") and decompress_gz
            final_path = output_path

            if is_gzip:
                _download_gzip(
                    output_path,
                    show_progress,
                    response,
                    total_size,
                    final_path,
                    description,
                )
            else:
                _download_nogzip(
                    output_path,
                    total_size,
                    response,
                    show_progress,
                    description,
                )
    return final_path


def get_file_last_modified(url: str, timeout: float = 30.0) -> datetime | None:
    """Get the Last-Modified date from a URL via HEAD request.

    Args:
        url: URL to check.
        timeout: Request timeout in seconds.

    Returns:
        The Last-Modified datetime or None if unavailable.
    """
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.head(url)
            if _is_cloudflare_blocked(response):
                raise CloudflareBlockedError(url)
            if "last-modified" in response.headers:
                from email.utils import parsedate_to_datetime

                return parsedate_to_datetime(response.headers["last-modified"])
    except (httpx.HTTPError, ValueError):
        pass
    return None


@contextmanager
def iter_with_progress(
    iterator: Iterable[bytes],
    *,
    enabled: bool,
    total: int | None,
    description: str,
) -> Iterator[Iterable[bytes]]:
    """Wrap an iterator with optional progress bar.

    Args:
        iterator: The byte iterator to wrap.
        enabled: Whether to show progress.
        total: Total size in bytes.
        description: Description for progress bar.

    Yields:
        The wrapped iterator.
    """
    if enabled and total and total > 0:
        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=description,
        ) as pbar:

            def gen() -> Generator[bytes, None, None]:
                """Yield chunks."""
                for chunk in iterator:
                    pbar.update(len(chunk))
                    yield chunk

            yield gen()
    else:
        yield iterator


def _download_gzip(
    output_path: Path,
    show_progress: bool,
    response: httpx.Response,
    total_size: int,
    final_path: Path,
    description: str | None = None,
) -> None:
    """Help download gzipped files.

    Args:
        output_path: Path for the output file.
        show_progress: Whether to show progress bar.
        response: HTTP response object.
        total_size: Total size in bytes.
        final_path: Final destination path.
        description: Description for progress bar.
    """
    temp_path = output_path.with_suffix(output_path.suffix + ".gz")

    with temp_path.open("wb") as f:
        with iter_with_progress(
            response.iter_bytes(chunk_size=8192),
            enabled=show_progress,
            total=total_size,
            description=f"Downloading {description}",
        ) as chunks:
            for chunk in chunks:
                f.write(chunk)

    if show_progress:
        compressed_size = temp_path.stat().st_size
    else:
        compressed_size = None

    with gzip.open(temp_path, "rb") as f_in, final_path.open("wb") as f_out:
        with iter_with_progress(
            iter(lambda: f_in.read(8192), b""),
            enabled=show_progress,
            total=compressed_size,
            description=f"Decompressing {description}",
        ) as chunks:
            for chunk in chunks:
                f_out.write(chunk)

    temp_path.unlink()


def _download_nogzip(
    output_path: Path,
    total_size: int,
    response: httpx.Response,
    show_progress: bool,
    description: str | None,
) -> None:
    """Help download non-gzipped files.

    Args:
        output_path: Path for the output file.
        total_size: Total size in bytes.
        response: HTTP response object.
        show_progress: Whether to show progress bar.
        description: Description for progress bar.
    """
    with output_path.open("wb") as f:
        with iter_with_progress(
            response.iter_bytes(chunk_size=8192),
            enabled=show_progress,
            total=total_size,
            description=f"Downloading {description}",
        ) as chunks:
            for chunk in chunks:
                f.write(chunk)


class _HasDownloadUrls(Protocol):
    """Structural type for a datasource config: just enough for URL/date resolution."""

    name: str
    download_urls: dict[str, Any]


class _HasListVersions(Protocol):
    """Structural type for a downloader class: just enough to list archive versions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Construct a downloader instance."""
        ...

    def list_versions(self) -> list[str]:
        """List all available archive versions for this datasource."""
        ...


#: Per-datasource hook resolving download URLs and the upstream release date
#: for a given version (``None`` meaning "latest"). Every entry shares this
#: signature regardless of what each datasource actually needs from
#: ``**kwargs`` (e.g. ``subset`` for ChEBI, ``species`` for Ensembl).
UrlsAndDate = Callable[..., tuple[dict[str, str], datetime | None]]


def _default_urls_and_date(
    version: str | None, config: _HasDownloadUrls, **kwargs: Any
) -> tuple[dict[str, str], datetime | None]:
    """Fall back for datasources with no registered resolver: latest config URLs, no date."""
    if version:
        logger.warning(
            "%s does not have versioned archives. Downloading latest version instead.",
            config.name,
        )
    return dict(config.download_urls), None


def resolve_datasource_urls(
    datasource_lower: str,
    config: _HasDownloadUrls,
    version: str | None = None,
    *,
    urls_and_date: Mapping[str, UrlsAndDate],
    **kwargs: Any,
) -> tuple[dict[str, str], datetime | None]:
    """Resolve download URLs and the source release date for a datasource.

    The release date drives the SSSOM ``mapping_date`` of generated mapping
    sets, so it should reflect when the upstream data was released rather
    than when it was downloaded. *urls_and_date* supplies each datasource's
    own most-specific resolution logic; when *datasource_lower* has no entry,
    :func:`_default_urls_and_date` is used. Either way, a generic HTTP
    ``Last-Modified`` lookup on the first URL is the final fallback when no
    release date could be determined.

    Args:
        datasource_lower: Lowercase datasource name.
        config: Datasource configuration.
        version: Specific version to get URLs for.
        urls_and_date: ``{datasource_name: resolver}`` registry.
        **kwargs: Forwarded to the resolved per-datasource hook.

    Returns:
        Tuple of (file-key -> URL mapping, release date or None).
    """
    resolver: UrlsAndDate = urls_and_date.get(datasource_lower, _default_urls_and_date)
    urls, release_date = resolver(version, config, **kwargs)

    # Generic fallback: derive the release date from the file's Last-Modified
    # header when a more specific signal was not available above. Must never
    # raise: some sources (e.g. HMDB) sit behind Cloudflare and block even
    # this lightweight HEAD request, and a date-resolution failure must not
    # break the actual file download.
    if release_date is None and urls:
        first_url = next(iter(urls.values()))
        try:
            release_date = get_file_last_modified(first_url)
        except CloudflareBlockedError:
            logger.debug("Could not resolve release date for %s: Cloudflare-blocked.", first_url)

    return urls, release_date


def get_download_urls(
    datasource: str,
    version: str | None = None,
    *,
    all_datasources: Mapping[str, _HasDownloadUrls],
    urls_and_date: Mapping[str, UrlsAndDate],
    **kwargs: Any,
) -> dict[str, str]:
    """Get download URLs for a datasource.

    Args:
        datasource: Name of the datasource.
        version: Specific version to get URLs for.
        all_datasources: ``{datasource_name: config}`` registry.
        urls_and_date: ``{datasource_name: resolver}`` registry, see
            :func:`resolve_datasource_urls`.
        **kwargs: Datasource-specific knobs forwarded to the resolved hook.

    Returns:
        Dictionary mapping file keys to URLs.
    """
    datasource_lower = datasource.lower()
    config = all_datasources.get(datasource_lower)
    if not config:
        raise ValueError(f"Unknown datasource: {datasource}")
    urls, _ = resolve_datasource_urls(
        datasource_lower, config, version, urls_and_date=urls_and_date, **kwargs
    )
    return urls


def resolve_release_date(
    datasource: str,
    version: str | None = None,
    *,
    all_datasources: Mapping[str, _HasDownloadUrls],
    urls_and_date: Mapping[str, UrlsAndDate],
    **kwargs: Any,
) -> datetime | None:
    """Resolve the upstream release date for a datasource/version.

    This is the date used for the SSSOM ``mapping_date`` of generated mapping
    sets. It does not download the data files (it may issue a lightweight
    ``HEAD`` request to read a ``Last-Modified`` header). Prefer
    :func:`download_datasource_with_release` when you are downloading
    anyway, to avoid an extra round-trip.

    Args:
        datasource: Name of the datasource.
        version: Specific version, when applicable.
        all_datasources: ``{datasource_name: config}`` registry.
        urls_and_date: ``{datasource_name: resolver}`` registry, see
            :func:`resolve_datasource_urls`.
        **kwargs: Datasource-specific knobs; see :func:`get_download_urls`.

    Returns:
        The release date, or None when it cannot be determined.
    """
    datasource_lower = datasource.lower()
    config = all_datasources.get(datasource_lower)
    if not config:
        raise ValueError(f"Unknown datasource: {datasource}")
    _, release_date = resolve_datasource_urls(
        datasource_lower, config, version, urls_and_date=urls_and_date, **kwargs
    )
    return release_date


def get_latest_release_info(
    datasource: str, *, checkers: Mapping[str, Callable[[], ReleaseInfo]]
) -> ReleaseInfo:
    """Get release information for a datasource.

    Args:
        datasource: Name of the datasource.
        checkers: ``{datasource_name: check_release_fn}`` registry.

    Returns:
        ReleaseInfo with the latest release details.

    Raises:
        ValueError: If the datasource is not supported.
    """
    checker = checkers.get(datasource.lower())
    if checker is None:
        raise ValueError(f"Unknown datasource: {datasource}. Supported: {sorted(checkers)}")
    return checker()


def check_release(
    datasource: str,
    current_version: str | None = None,
    current_date: datetime | None = None,
    *,
    checkers: Mapping[str, Callable[[], ReleaseInfo]],
) -> ReleaseInfo:
    """Check if a new release is available for a datasource.

    Args:
        datasource: Name of the datasource.
        current_version: Current version string to compare against.
        current_date: Current release date to compare against.
        checkers: ``{datasource_name: check_release_fn}`` registry.

    Returns:
        ReleaseInfo with is_new indicating if update is available.
    """
    info = get_latest_release_info(datasource, checkers=checkers)

    if current_version and info.version:
        info = ReleaseInfo(
            datasource=info.datasource,
            version=info.version,
            release_date=info.release_date,
            is_new=info.version != current_version,
            files=info.files,
        )
    elif current_date and info.release_date:
        info = ReleaseInfo(
            datasource=info.datasource,
            version=info.version,
            release_date=info.release_date,
            is_new=info.release_date > current_date,
            files=info.files,
        )

    return info


def list_versions(
    datasource: str,
    *,
    known_datasources: Iterable[str],
    downloaders: Mapping[str, type[_HasListVersions]],
) -> list[str]:
    """List all available archive versions for a datasource.

    Delegates to the datasource's downloader class ``list_versions()``
    method, which contains all source-specific retrieval logic.

    Args:
        datasource: Datasource name.
        known_datasources: Every datasource name this package supports
            (used only to distinguish "unknown datasource" from "known
            datasource with no versioned archive").
        downloaders: ``{datasource_name: downloader_class}`` registry;
            datasources with no versioned archive are simply absent.

    Returns:
        Sorted list of version strings available for download.

    Raises:
        ValueError: If the datasource is unknown or has no versioned archive.
    """
    lower = datasource.lower()
    known_lower = {d.lower() for d in known_datasources}
    if lower not in known_lower:
        raise ValueError(f"Unknown datasource: {datasource!r}. Supported: {sorted(known_lower)}")

    cls = downloaders.get(lower)
    if cls is None:
        raise ValueError(
            f"{datasource.upper()} does not maintain a versioned archive. "
            "Only the latest release is available for download."
        )
    return cls().list_versions()


def download_datasource_with_release(
    datasource: str,
    output_dir: Path,
    *,
    all_datasources: Mapping[str, _HasDownloadUrls],
    urls_and_date: Mapping[str, UrlsAndDate],
    decompress: bool = True,
    version: str | None = None,
    keys: list[str] | None = None,
    tar_extractors: Mapping[str, Callable[[Path, Path], dict[str, Path]]] | None = None,
    show_progress: bool = True,
    **kwargs: Any,
) -> tuple[dict[str, Path], datetime | None]:
    """Download all files for a datasource and report its release date.

    Args:
        datasource: Name of the datasource.
        output_dir: Directory to save files.
        all_datasources: ``{datasource_name: config}`` registry.
        urls_and_date: ``{datasource_name: resolver}`` registry, see
            :func:`resolve_datasource_urls`.
        decompress: Whether to decompress .gz files.
        version: Specific version to download. Format depends on datasource.
        keys: Optional list of file-key names to download.
        tar_extractors: ``{datasource_name: extractor}`` registry for
            datasources that publish a ``.tar.gz`` archive needing
            member-level extraction.
        show_progress: Whether to show download/decompression progress bars.
        **kwargs: Datasource-specific knobs forwarded to the resolved hook.

    Returns:
        Tuple of (file-key -> downloaded path mapping, release date or None).
    """
    datasource_lower = datasource.lower()
    config = all_datasources.get(datasource_lower)
    if not config:
        raise ValueError(f"Unknown datasource: {datasource}")

    output_dir.mkdir(parents=True, exist_ok=True)
    urls, release_date = resolve_datasource_urls(
        datasource_lower, config, version, urls_and_date=urls_and_date, **kwargs
    )
    if keys is not None:
        urls = {k: v for k, v in urls.items() if k in keys}

    extractor = (tar_extractors or {}).get(datasource_lower)
    return (
        download_urls(
            urls, output_dir, decompress, tar_extractor=extractor, show_progress=show_progress
        ),
        release_date,
    )


def download_datasource(
    datasource: str,
    output_dir: Path,
    *,
    all_datasources: Mapping[str, _HasDownloadUrls],
    urls_and_date: Mapping[str, UrlsAndDate],
    decompress: bool = True,
    version: str | None = None,
    keys: list[str] | None = None,
    tar_extractors: Mapping[str, Callable[[Path, Path], dict[str, Path]]] | None = None,
    show_progress: bool = True,
    **kwargs: Any,
) -> dict[str, Path]:
    """Download all files for a datasource.

    Same as :func:`download_datasource_with_release`, but discards the
    resolved release date.

    Returns:
        Dictionary mapping file keys to downloaded paths.
    """
    files, _ = download_datasource_with_release(
        datasource,
        output_dir,
        all_datasources=all_datasources,
        urls_and_date=urls_and_date,
        decompress=decompress,
        version=version,
        keys=keys,
        tar_extractors=tar_extractors,
        show_progress=show_progress,
        **kwargs,
    )
    return files


def download_urls(
    urls: dict[str, str],
    output_dir: Path,
    decompress: bool = True,
    *,
    tar_extractor: Callable[[Path, Path], dict[str, Path]] | None = None,
    show_progress: bool = True,
) -> dict[str, Path]:
    """Download files from URLs to output directory.

    Args:
        urls: Dictionary mapping file keys to URLs.
        output_dir: Directory to save files.
        decompress: Whether to decompress .gz files.
        tar_extractor: When given, any ``.tar.gz`` URL is downloaded then
            passed to this callable for member-level extraction instead of
            being treated as a single downloaded file.
        show_progress: Whether to show download/decompression progress bars.

    Returns:
        Dictionary mapping file keys to downloaded paths.
    """
    downloaded: dict[str, Path] = {}

    for key, url in urls.items():
        filename = url.split("/")[-1]

        if tar_extractor is not None and filename.endswith(".tar.gz"):
            output_path = output_dir / filename
            logger.info("Downloading %s: %s", key, url)
            download_file(url, output_path, decompress_gz=False, show_progress=show_progress)
            extracted = tar_extractor(output_path, output_dir)
            downloaded.update(extracted)
            logger.info("Extracted: %s", list(extracted.keys()))
            continue

        if decompress and filename.endswith(".gz"):
            filename = filename[:-3]

        output_path = output_dir / filename
        logger.info("Downloading %s: %s", key, url)
        download_file(url, output_path, decompress_gz=decompress, show_progress=show_progress)
        downloaded[key] = output_path
        logger.info("Saved to: %s", output_path)

    return downloaded
