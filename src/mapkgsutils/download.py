"""Generic file-download primitives shared by every datasource downloader."""

from __future__ import annotations

import gzip
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from tqdm import tqdm

__all__ = [
    "CloudflareBlockedError",
    "ReleaseInfo",
    "download_file",
    "get_file_last_modified",
    "iter_with_progress",
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
