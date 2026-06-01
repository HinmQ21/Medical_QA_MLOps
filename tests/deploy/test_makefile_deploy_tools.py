from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_makefile_exposes_deploy_targets_and_preserves_pipeline_targets():
    text = (ROOT / "Makefile").read_text()
    for target in [
        "install-deploy-tools:",
        "helm-lint:",
        "helm-template:",
        "helm-dry-run:",
        "docker-build:",
        "dvc-status:",
    ]:
        assert target in text
    assert ".venv/bin/dvc status" in text
    assert "deploy/helm/api" in text
    assert "deploy/helm/retrieval" in text
    assert "deploy/helm/nginx" in text
    assert "deploy/helm/kserve" in text
    assert "docker/api.Dockerfile" in text
    assert "docker/retrieval.Dockerfile" in text
    assert "docker/kserve-mock.Dockerfile" in text
    assert "docker/pipeline-init.Dockerfile" in text


def test_deploy_tool_installer_pins_helm_and_kubectl():
    text = (ROOT / "scripts/install_deploy_tools.py").read_text()
    assert 'HELM_VERSION = "v3.15.4"' in text
    assert 'KUBECTL_VERSION = "v1.30.4"' in text
    assert "get.helm.sh" in text
    assert "dl.k8s.io" in text
