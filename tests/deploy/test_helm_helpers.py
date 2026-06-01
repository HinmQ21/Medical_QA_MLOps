from __future__ import annotations

import subprocess

import pytest

from tests.deploy import helm_helpers


def test_render_chart_reports_helm_stdout_and_stderr(monkeypatch, tmp_path):
    helm = tmp_path / "helm"
    helm.touch()
    monkeypatch.setattr(helm_helpers, "HELM", helm)

    def fail_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=cmd,
            output="rendered output",
            stderr="template error",
        )

    monkeypatch.setattr(helm_helpers.subprocess, "run", fail_run)

    with pytest.raises(AssertionError) as exc_info:
        helm_helpers.render_chart("rag-api")

    message = str(exc_info.value)
    assert "helm template failed" in message
    assert "rendered output" in message
    assert "template error" in message
    assert str(helm) in message


def test_render_chart_sets_timeout(monkeypatch, tmp_path):
    helm = tmp_path / "helm"
    helm.touch()
    monkeypatch.setattr(helm_helpers, "HELM", helm)

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(helm_helpers.subprocess, "run", fake_run)

    helm_helpers.render_chart("rag-api")

    assert calls[0][1]["timeout"] == 30


def test_find_kind_error_handles_missing_or_invalid_metadata():
    resources = [
        {"kind": "Service"},
        {"kind": "Deployment", "metadata": "api"},
        {"kind": "ConfigMap", "metadata": {"name": "settings"}},
    ]

    with pytest.raises(AssertionError) as exc_info:
        helm_helpers.find_kind(resources, "Secret", "token")

    assert str(exc_info.value) == (
        "missing Secret/token; available="
        "['Service/<unknown>', 'Deployment/<unknown>', 'ConfigMap/settings']"
    )
