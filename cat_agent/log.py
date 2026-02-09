"""Logging configuration for cat-agent (powered by `loguru`_).

By default the logger is silent (all sinks removed).
Set the ``CAT_AGENT_LOG_LEVEL`` environment variable to activate console output::

    CAT_AGENT_LOG_LEVEL=DEBUG python my_script.py

Environment variables
---------------------
``CAT_AGENT_LOG_LEVEL``
    Logging verbosity: ``TRACE``, ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``,
    or ``CRITICAL``.  When unset, the logger produces no output (library-safe).

``CAT_AGENT_LOG_FILE``
    Optional path to a log file.  Rotation is set to 10 MB with 5 back-ups
    and 30-day retention.

``CAT_AGENT_LOG_FORMAT``
    ``pretty`` (default) -- human-readable coloured output.
    ``json`` -- structured JSON serialisation (one object per line).

Programmatic usage
------------------
>>> from cat_agent.log import setup_logger
>>> setup_logger(level="DEBUG")                         # coloured stderr
>>> setup_logger(level="INFO", log_file="/tmp/cat.log") # + rotating file
>>> setup_logger(level="DEBUG", fmt="json")             # structured JSON
"""

from __future__ import annotations

import os
import sys
from typing import Literal

from loguru import logger

# Re-export so every module can keep doing ``from cat_agent.log import logger``
__all__ = ["logger", "setup_logger"]

# ---------------------------------------------------------------------------
# Format strings
# ---------------------------------------------------------------------------

_PRETTY_FORMAT = (
    "<dim>{time:YYYY-MM-DD HH:mm:ss.SSS}</dim> "
    "<level>{level:<8}</level> "
    "<dim>{name}:{function}:{line}</dim> "
    "<dim>│</dim> {message}"
)

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} "
    "{level:<8} "
    "{name}:{function}:{line} "
    "│ {message}"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logger(
    level: str | None = None,
    *,
    log_file: str | None = None,
    fmt: Literal["pretty", "json"] = "pretty",
) -> None:
    """(Re)configure the ``cat_agent`` logger.

    Parameters
    ----------
    level:
        Minimum severity to emit.  When *None* the logger is fully silent
        (all sinks removed) -- safe for library consumers who configure
        their own logging.
    log_file:
        Optional path to a log file.  Uses loguru's built-in rotation
        (10 MB), retention (30 days) and compression (gz).
    fmt:
        ``"pretty"`` for coloured, human-readable output (default).
        ``"json"`` for structured JSON serialisation.
    """
    # Remove every existing sink (including the default stderr one).
    logger.remove()

    if level is None:
        # Library-safe: produce no output at all.
        logger.disable("cat_agent")
        return

    # Make sure loguru is enabled for our namespace.
    logger.enable("cat_agent")

    serialize = fmt == "json"

    # ---- stderr sink ----
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=_PRETTY_FORMAT if not serialize else "{message}",
        serialize=serialize,
        colorize=True,
        backtrace=True,
        diagnose=False,  # avoid leaking local vars in prod
    )

    # ---- optional file sink ----
    if log_file:
        logger.add(
            log_file,
            level=level.upper(),
            format=_FILE_FORMAT if not serialize else "{message}",
            serialize=serialize,
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            encoding="utf-8",
            backtrace=True,
            diagnose=False,
        )


# ---------------------------------------------------------------------------
# Module-level initialisation
# ---------------------------------------------------------------------------

# Remove loguru's default stderr sink immediately; we only add sinks when
# explicitly configured (env var or programmatic call).
logger.remove()

_env_level = os.environ.get("CAT_AGENT_LOG_LEVEL")
_env_file = os.environ.get("CAT_AGENT_LOG_FILE")
_env_fmt: Literal["pretty", "json"] = (
    "json" if os.environ.get("CAT_AGENT_LOG_FORMAT", "").lower() == "json" else "pretty"
)

if _env_level:
    setup_logger(level=_env_level, log_file=_env_file, fmt=_env_fmt)
else:
    # Fully silent by default -- library-friendly.
    logger.disable("cat_agent")
