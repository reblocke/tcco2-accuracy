"""Output-boundary guard for restricted-data-derived workflow artifacts."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_ALLOWED_REPOSITORY_ROOTS = frozenset({".pytest_tmp", ".tmp"})


def require_private_output_path(path: Path | str) -> Path:
    """Return a resolved output path after enforcing the private-output boundary.

    Restricted-data-derived outputs may be written within this repository only
    below ``.pytest_tmp/`` or ``.tmp/``. Explicit paths outside the repository
    remain valid private-workspace destinations.
    """

    resolved = Path(path).expanduser().resolve()
    try:
        repository_relative = resolved.relative_to(REPOSITORY_ROOT)
    except ValueError:
        return resolved

    if repository_relative.parts and repository_relative.parts[0] in _ALLOWED_REPOSITORY_ROOTS:
        return resolved

    allowed = ", ".join(f"{name}/" for name in sorted(_ALLOWED_REPOSITORY_ROOTS))
    raise ValueError(
        "Restricted-data-derived outputs cannot be written to a tracked repository "
        f"location: {resolved}. Use {allowed} or an explicit path outside the repository."
    )
