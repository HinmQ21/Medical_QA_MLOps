import hashlib
import io
import importlib.util
import subprocess
import sys
import tarfile
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


def test_makefile_uses_file_targets_for_deploy_tools_and_pipefail():
    text = (ROOT / "Makefile").read_text()
    assert "SHELL := /bin/bash" in text
    assert ".SHELLFLAGS := -eu -o pipefail -c" in text
    assert "install-deploy-tools: $(HELM) $(KUBECTL)" in text
    assert "$(HELM) $(KUBECTL) &: scripts/install_deploy_tools.py | .venv" in text
    assert "helm-lint: $(HELM) $(KUBECTL)" in text
    assert "helm-template: $(HELM) $(KUBECTL)" in text
    assert "helm-dry-run: $(HELM) $(KUBECTL)" in text
    assert "helm-lint: install-deploy-tools" not in text
    assert "helm-template: install-deploy-tools" not in text
    assert "helm-dry-run: install-deploy-tools" not in text


def test_deploy_tool_installer_pins_helm_and_kubectl():
    text = (ROOT / "scripts/install_deploy_tools.py").read_text()
    assert 'HELM_VERSION = "v3.15.4"' in text
    assert 'KUBECTL_VERSION = "v1.30.4"' in text
    assert "get.helm.sh" in text
    assert "dl.k8s.io" in text
    assert ".sha256sum" in text
    assert ".sha256" in text


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


def test_kubectl_install_rejects_checksum_mismatch(monkeypatch, tmp_path):
    installer = _load_installer()

    def fake_download(url, destination):
        if url.endswith(".sha256"):
            destination.write_text("0" * 64)
        else:
            destination.write_bytes(b"kubectl")

    monkeypatch.setattr(installer, "_download", fake_download)

    try:
        installer._install_kubectl(tmp_path, "amd64")
    except RuntimeError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("kubectl install accepted a mismatched checksum")


def test_helm_install_verifies_checksum_and_extracts_only_expected_member(
    monkeypatch, tmp_path
):
    installer = _load_installer()
    archive_bytes = _helm_archive_bytes("linux-amd64/helm", b"helm-binary")
    expected_checksum = hashlib.sha256(archive_bytes).hexdigest()

    def fake_download(url, destination):
        if url.endswith(".sha256sum"):
            destination.write_text(f"{expected_checksum}  helm.tar.gz\n")
        else:
            destination.write_bytes(archive_bytes)

    def fail_extractall(self, *args, **kwargs):
        raise AssertionError("helm installer must not call extractall")

    monkeypatch.setattr(installer, "_download", fake_download)
    monkeypatch.setattr(installer.tarfile.TarFile, "extractall", fail_extractall)

    installer._install_helm(tmp_path, "amd64")

    helm = tmp_path / "helm"
    assert helm.read_bytes() == b"helm-binary"
    assert helm.stat().st_mode & 0o111


def _helm_archive_bytes(member_name, content):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        data = io.BytesIO(content)
        info = tarfile.TarInfo(member_name)
        info.size = len(content)
        archive.addfile(info, data)
    return buffer.getvalue()
