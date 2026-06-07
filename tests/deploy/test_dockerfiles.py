from pathlib import Path


ROOT = Path(__file__).parents[2]


def _read(name: str) -> str:
    return (ROOT / "docker" / name).read_text()


def test_dockerignore_excludes_local_and_generated_build_context():
    text = (ROOT / ".dockerignore").read_text()
    ignored = [
        ".git/",
        ".venv/",
        ".tools/",
        ".worktrees/",
        ".dvc/cache/",
        ".dvc/tmp/",
        ".dvc/config.local",
        "artifacts/",
        "__pycache__/",
        "*.py[cod]",
        "*.egg-info/",
        ".pytest_cache/",
        ".coverage",
    ]
    for needle in ignored:
        assert needle in text


def test_api_dockerfile_is_lean_non_root_and_binds_all_interfaces():
    text = _read("api.Dockerfile")
    assert "FROM python:3.12-slim" in text
    assert "uv pip install --system --no-cache ." in text
    assert ".[runtime]" not in text
    assert ".[pipeline]" not in text
    assert "USER app" in text
    assert "EXPOSE 8000" in text
    assert '"--host", "0.0.0.0"' in text
    assert '"--port", "8000"' in text
    assert "medical_qa_platform.api.app:create_app" in text


def test_retrieval_dockerfile_installs_runtime_only_and_does_not_bake_artifacts():
    text = _read("retrieval.Dockerfile")
    assert "uv pip install --system --no-cache '.[runtime]'" in text
    assert "download.pytorch.org/whl/cpu" in text  # CPU-only torch, not the ~2.5GB CUDA build
    assert "uv pip install" in text
    assert "USER app" in text
    assert "EXPOSE 8001" in text
    assert "RETRIEVAL_DEVICE=cpu" in text
    assert "HF_HOME=/mnt/artifacts/hf" in text
    assert "SENTENCE_TRANSFORMERS_HOME=/mnt/artifacts/hf/sentence-transformers" in text
    assert '"--host", "0.0.0.0"' in text
    assert '"--port", "8001"' in text
    forbidden = ["baseline", "medical_hg.json", "index_hyperedge.bin", "index_entity.bin", "hedge_ids.npy", "entity_names.npy"]
    for needle in forbidden:
        assert needle not in text


def test_kserve_mock_dockerfile_runs_mock_predictor_on_8080():
    text = _read("kserve-mock.Dockerfile")
    assert "uv pip install --system --no-cache ." in text
    assert "USER app" in text
    assert "EXPOSE 8080" in text
    assert "medical_qa_platform.serving.kserve_mock_app:create_app" in text
    assert '"--host", "0.0.0.0"' in text
    assert '"--port", "8080"' in text


def test_pipeline_init_dockerfile_contains_dvc_but_not_runtime_ml_dependencies():
    text = _read("pipeline-init.Dockerfile")
    assert "dvc[gs]>=3.50" in text  # gs extra: keyless GCS pull in the dvc-pull initContainer
    assert ".[runtime]" not in text
    assert ".[pipeline]" not in text
    assert "mlflow" not in text
    assert "COPY src" not in text
    assert "COPY mlops" not in text
    assert "USER app" in text
    assert 'CMD ["dvc", "--version"]' in text


def test_ui_dockerfile_runs_streamlit_on_8501_non_root():
    text = _read("ui.Dockerfile")
    assert "FROM python:3.12-slim" in text
    assert "uv pip install --system --no-cache '.[demo]'" in text
    assert "COPY src ./src" in text  # .[demo] builds medical_qa_platform from src/
    assert "COPY app ./app" in text
    assert "USER app" in text
    assert "EXPOSE 8501" in text
    assert '"streamlit", "run", "app/streamlit_app.py"' in text
    assert '"--server.port", "8501"' in text
    assert '"--server.address", "0.0.0.0"' in text
