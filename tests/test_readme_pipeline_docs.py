from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_readme_documents_smoke_pipeline_commands():
    text = (ROOT / "README.md").read_text()
    assert "make install-pipeline" in text
    assert "make smoke-pipeline-local" in text
    assert "make smoke-pipeline" in text
    assert "make mlflow-register-dry-run" in text
    assert "DVC" in text
    assert "MLflow" in text
