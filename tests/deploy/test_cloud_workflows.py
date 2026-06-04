from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
WF = ROOT / ".github/workflows"


def _load(name: str) -> dict:
    return yaml.safe_load((WF / name).read_text())


def _triggers(wf: dict):
    # PyYAML parses the bare key `on:` as the boolean True.
    return wf.get("on", wf.get(True))


def test_demo_up_is_dispatch_and_uses_oidc_and_scripts():
    text = (WF / "demo-up.yml").read_text()
    wf = _load("demo-up.yml")
    assert "workflow_dispatch" in _triggers(wf)
    assert "id-token: write" in text                       # OIDC token permission
    assert "google-github-actions/auth@v2" in text
    assert "workload_identity_provider: ${{ vars.GCP_WIF_PROVIDER }}" in text
    assert "service_account: ${{ vars.GCP_DEPLOY_SA }}" in text
    assert "google-github-actions/get-gke-credentials@v2" in text
    assert "scripts/cloud/provision_gke.sh" in text
    assert "scripts/cloud/create_secrets.sh" in text
    assert "scripts/cloud/deploy.sh" in text
    assert "scripts/cloud/smoke_cloud.sh" in text
