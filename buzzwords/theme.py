"""Loads the app stylesheet for launch(css=...)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_CSS = Path(__file__).resolve().parent / "static" / "styles.css"


@lru_cache(maxsize=1)
def get_css() -> str:
    return _CSS.read_text(encoding="utf-8") if _CSS.exists() else ""
