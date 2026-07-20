"""Minimal SPARQL SELECT client, used for datasource-version lookups."""

from __future__ import annotations

import httpx

__all__ = ["query_sparql_scalar"]


def query_sparql_scalar(query: str, endpoint: str, timeout: float = 30.0) -> str | None:
    """Run a SPARQL SELECT and return its first result's first binding.

    Meant for small lookups, e.g. a datasource's current version/release
    date, rather than bulk data retrieval.

    Args:
        query: SPARQL SELECT query.
        endpoint: SPARQL endpoint URL to query.
        timeout: Request timeout in seconds.

    Returns:
        The value of the first binding in the first result row, or ``None``
        if the query returned no rows.
    """
    response = httpx.get(
        endpoint,
        params={"query": query},
        headers={"Accept": "application/sparql-results+json"},
        timeout=timeout,
    )
    response.raise_for_status()
    bindings = response.json().get("results", {}).get("bindings", [])
    if not bindings:
        return None
    first_binding = next(iter(bindings[0].values()), None)
    return first_binding.get("value") if first_binding else None
