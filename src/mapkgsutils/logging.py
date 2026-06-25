"""Logging configuration for mapkgsutils.

This module provides a centralized logger for the package.
By default, only CRITICAL messages are shown. Use set_log_level()
to adjust verbosity.
"""

from __future__ import annotations

import logging
import sys

__all__ = [
    "LOG_LEVELS",
    "logger",
    "set_log_level",
]

# Package logger
logger = logging.getLogger("mapkgsutils")

# Default to CRITICAL (minimal output)
logger.setLevel(logging.CRITICAL)

# Add a handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)

# Map of log level names to logging constants
LOG_LEVELS: dict[str, int] = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def set_log_level(level: str | int) -> None:
    """Set the logging level for mapkgsutils.

    Args:
        level: Log level as string ('debug', 'info', 'warning', 'error', 'critical')
               or as an int (logging.DEBUG, logging.INFO, etc.)

    Example:
        >>> from mapkgsutils.logging import set_log_level
        >>> set_log_level("warning")  # Show warnings and above
        >>> set_log_level("debug")  # Show all messages
    """
    if isinstance(level, str):
        level_int = LOG_LEVELS.get(level.lower())
        if level_int is None:
            raise ValueError(f"Unknown log level: {level}. Available: {list(LOG_LEVELS.keys())}")
        logger.setLevel(level_int)
    else:
        logger.setLevel(level)
