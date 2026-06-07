import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_demo_extra_declares_streamlit_and_httpx():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    demo = "\n".join(data["project"]["optional-dependencies"]["demo"])
    assert "streamlit" in demo
    assert "httpx" in demo


def test_makefile_exposes_demo_ui_target():
    text = (ROOT / "Makefile").read_text()
    assert "demo-ui:" in text
    assert "streamlit run app/streamlit_app.py" in text


def test_streamlit_app_is_render_only_and_imports_client():
    text = (ROOT / "app/streamlit_app.py").read_text()
    assert "from app.client import" in text
    assert "import streamlit" in text
    # render-only: business logic must live in app.client, not be duplicated here
    assert "def predict(" not in text
    assert "def fetch_version(" not in text
    assert "class PredictResult" not in text
