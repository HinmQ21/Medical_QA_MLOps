from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).parents[2]
HELM = ROOT / ".tools/bin/helm"


def require_helm() -> Path:
    if not HELM.exists():
        pytest.skip("project-local helm is not installed; run make install-deploy-tools")
    return HELM


def render_chart(chart_name: str) -> list[dict]:
    helm = require_helm()
    chart = ROOT / "deploy/helm" / chart_name
    cmd = [str(helm), "template", f"test-{chart_name}", str(chart)]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout if exc.stdout is not None else exc.output
        raise AssertionError(
            "helm template failed\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{stdout or ''}\n"
            f"stderr:\n{exc.stderr or ''}"
        ) from exc
    return [
        doc
        for doc in yaml.safe_load_all(result.stdout)
        if isinstance(doc, dict) and doc
    ]


def _resource_name(resource: dict) -> str:
    metadata = resource.get("metadata", {})
    if not isinstance(metadata, dict):
        return "<unknown>"
    return metadata.get("name") or "<unknown>"


def find_kind(resources: list[dict], kind: str, name: str | None = None) -> dict:
    for resource in resources:
        if resource.get("kind") != kind:
            continue
        if name is None or _resource_name(resource) == name:
            return resource
    available = [
        f"{resource.get('kind')}/{_resource_name(resource)}"
        for resource in resources
    ]
    raise AssertionError(f"missing {kind}/{name}; available={available}")
