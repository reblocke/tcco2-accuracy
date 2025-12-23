from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def test_streamlit_app_imports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    monkeypatch.chdir(tmp_path)
    root = Path(__file__).resolve().parents[2]
    app_path = root / "streamlit_app.py"
    if not app_path.exists():
        pytest.skip("Streamlit app not found.")
    spec = importlib.util.spec_from_file_location("streamlit_app", app_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
