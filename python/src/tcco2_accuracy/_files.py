"""Small file-writing helpers shared by workflow modules."""

from __future__ import annotations

from pathlib import Path


def write_text(path: Path, content: str) -> None:
    """Write text content, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
