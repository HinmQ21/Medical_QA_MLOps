from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]


def test_pipeline_extra_contains_dvc_mlflow_and_pyyaml():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    optional = data["project"]["optional-dependencies"]
    pipeline = "\n".join(optional["pipeline"])
    assert "dvc" in pipeline
    assert "mlflow" in pipeline
    assert "PyYAML" in pipeline


def test_dev_extra_contains_pyyaml_for_yaml_shape_tests():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dev = "\n".join(data["project"]["optional-dependencies"]["dev"])
    assert "PyYAML" in dev


def test_makefile_exposes_pipeline_targets():
    text = (ROOT / "Makefile").read_text()
    for target in [
        "install-pipeline:",
        "smoke-pipeline:",
        "smoke-pipeline-local:",
        "mlflow-register-dry-run:",
        "register-model:",
        "dvc-status:",
    ]:
        assert target in text
