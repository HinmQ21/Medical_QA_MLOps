import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]


def _load_installer():
    path = ROOT / "scripts/install_deploy_tools.py"
    spec = importlib.util.spec_from_file_location("install_deploy_tools", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_deploy_tool_installer_fails_when_verification_command_fails(
    monkeypatch, tmp_path
):
    installer = _load_installer()
    monkeypatch.setattr(installer, "install", lambda bin_dir: None)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    helm = bin_dir / "helm"
    kubectl = bin_dir / "kubectl"
    helm.write_text("#!/bin/sh\nexit 0\n")
    kubectl.write_text("#!/bin/sh\nexit 7\n")
    helm.chmod(0o755)
    kubectl.chmod(0o755)

    monkeypatch.setattr(
        sys, "argv", ["install_deploy_tools.py", "--bin-dir", str(bin_dir)]
    )

    try:
        installer.main()
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("installer.main() ignored a failed verification command")
