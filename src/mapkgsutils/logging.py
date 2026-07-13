"""Logging configuration.

Provides a configured package logger. Only CRITICAL messages show by default;
call :func:`set_log_level` to raise verbosity.
"""

from __future__ import annotations

import logging
import sys

__all__ = [
    "LOG_LEVELS",
    "get_logger",
    "logger",
    "set_log_level",
]

# Map of log level names to logging constants
LOG_LEVELS: dict[str, int] = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def get_logger(name: str) -> logging.Logger:
    """Return a stderr-handled logger for *name*, defaulting to CRITICAL."""
    log = logging.getLogger(name)
    log.setLevel(logging.CRITICAL)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        log.addHandler(handler)
    return log


logger = get_logger("mapkgsutils")


def set_log_level(level: str | int, target: logging.Logger | None = None) -> None:
    """Set the log level on *target* (the mapkgsutils logger by default).

    Args:
        level: Level name (``"debug"``/``"info"``/``"warning"``/``"error"``/
            ``"critical"``) or an integer such as ``logging.INFO``.
        target: Logger to configure.

    Example:
        >>> from mapkgsutils.logging import set_log_level
        >>> set_log_level("warning")
    """
    log = target if target is not None else logger
    if isinstance(level, str):
        level_int = LOG_LEVELS.get(level.lower())
        if level_int is None:
            raise ValueError(f"Unknown log level: {level}. Available: {list(LOG_LEVELS.keys())}")
        log.setLevel(level_int)
    else:
        log.setLevel(level)
