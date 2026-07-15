"""Load Cat-Agent configuration from a .env file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

_ENV_LOADED = False


def load_env_file(*, path: str | Path | None = None) -> bool:
    """Load environment variables from a dotenv file.

    Existing process environment variables are not overwritten.
    Returns True when a file was found and loaded.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return False

    from dotenv import load_dotenv

    for candidate in _candidate_paths(path):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            _ENV_LOADED = True
            return True

    _ENV_LOADED = True
    return False


def _candidate_paths(explicit: str | Path | None = None) -> Iterable[Path]:
    if explicit:
        yield Path(explicit)
        return

    env_file = os.getenv('CAT_AGENT_ENV_FILE', '').strip()
    if env_file:
        yield Path(env_file)
        return

    yield Path.cwd() / '.env'


def reset_env_loading() -> None:
    """Reset the load guard. Intended for tests only."""
    global _ENV_LOADED
    _ENV_LOADED = False
