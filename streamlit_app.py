"""Streamlit entrypoint for community cloud deployment."""

from pathlib import Path
import sys

# Ensure repo root is importable when pytest loads this file directly.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.streamlit_app import main


if __name__ == "__main__":
    main()
