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
    result = subprocess.run(
        [str(helm), "template", f"test-{chart_name}", str(chart)],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        doc
        for doc in yaml.safe_load_all(result.stdout)
        if isinstance(doc, dict) and doc
    ]


def find_kind(resources: list[dict], kind: str, name: str | None = None) -> dict:
    for resource in resources:
        if resource.get("kind") != kind:
            continue
        if name is None or resource.get("metadata", {}).get("name") == name:
            return resource
    available = [
        f"{resource.get('kind')}/{resource.get('metadata', {}).get('name')}"
        for resource in resources
    ]
    raise AssertionError(f"missing {kind}/{name}; available={available}")
