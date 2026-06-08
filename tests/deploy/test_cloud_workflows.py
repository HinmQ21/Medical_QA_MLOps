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


def test_deploy_auto_uses_oidc_and_is_gated_on_ci_success():
    text = (WF / "deploy.yml").read_text()
    assert "id-token: write" in text
    assert "google-github-actions/auth@v2" in text
    assert "workflow_run.conclusion == 'success'" in text
    assert "steps.cluster.outputs.up == 'true'" in text
    assert "google-github-actions/get-gke-credentials@v2" in text


def test_demo_down_is_dispatch_tears_down_and_optionally_deletes_bucket():
    text = (WF / "demo-down.yml").read_text()
    wf = _load("demo-down.yml")
    assert "workflow_dispatch" in _triggers(wf)
    assert "id-token: write" in text
    assert "google-github-actions/auth@v2" in text
    assert "scripts/cloud/teardown.sh" in text
    assert "delete_bucket" in text
    assert "inputs.delete_bucket == 'true'" in text
    assert "storage rm -r" in text


def test_demo_up_wires_llm_backend_toggle():
    text = (WF / "demo-up.yml").read_text()
    wf = _load("demo-up.yml")
    inputs = _triggers(wf)["workflow_dispatch"]["inputs"]
    assert inputs["backend"]["type"] == "choice"
    assert inputs["backend"]["default"] == "llm"
    assert "llm" in inputs["backend"]["options"]
    assert "mock" in inputs["backend"]["options"]
    assert "MODEL_BACKEND: ${{ inputs.backend }}" in text
    assert "LLM_API_KEY: ${{ secrets.LLM_API_KEY }}" in text
    assert "LLM_BASE_URL: ${{ vars.LLM_BASE_URL }}" in text
    assert "LLM_MODEL: ${{ vars.LLM_MODEL }}" in text


def test_deploy_auto_wires_llm_and_ensures_llm_secret():
    text = (WF / "deploy.yml").read_text()
    assert "MODEL_BACKEND: ${{ vars.MODEL_BACKEND || 'llm' }}" in text
    assert "LLM_API_KEY: ${{ secrets.LLM_API_KEY }}" in text
    assert "LLM_BASE_URL: ${{ vars.LLM_BASE_URL }}" in text
    assert "LLM_MODEL: ${{ vars.LLM_MODEL }}" in text
    # auto-deploy must ensure the LLM secret exists before flipping to llm
    assert "scripts/cloud/create_secrets.sh" in text


def test_demo_up_has_llm_host_incluster_path():
    text = (WF / "demo-up.yml").read_text()
    wf = _load("demo-up.yml")
    inputs = _triggers(wf)["workflow_dispatch"]["inputs"]
    assert inputs["llm_host"]["type"] == "choice"
    assert inputs["llm_host"]["default"] == "dgx"
    assert set(inputs["llm_host"]["options"]) == {"dgx", "in-cluster"}
    # in-cluster topology branch
    assert "scripts/cloud/provision_gke_standard.sh" in text
    assert "scripts/cloud/install_kserve.sh" in text
    assert "medical-qa-kserve-predictor" in text
    assert "isvc/medical-qa-kserve" in text
    # the Autopilot path is preserved (guarded, not removed)
    assert "scripts/cloud/provision_gke.sh" in text
    # get-credentials is now topology-driven, not hardcoded to medical-qa
    assert "cluster_name: ${{ env.GKE_CLUSTER }}" in text
    assert "location: ${{ env.GKE_LOCATION }}" in text


def test_demo_down_has_llm_host_topology_teardown():
    text = (WF / "demo-down.yml").read_text()
    wf = _load("demo-down.yml")
    inputs = _triggers(wf)["workflow_dispatch"]["inputs"]
    assert inputs["llm_host"]["type"] == "choice"
    assert inputs["llm_host"]["default"] == "dgx"
    assert set(inputs["llm_host"]["options"]) == {"dgx", "in-cluster"}
    # tears down the cluster the topology selected
    assert 'get-credentials "$GKE_CLUSTER" --location "$GKE_LOCATION"' in text
    assert "scripts/cloud/teardown.sh" in text
    # bucket-delete escape hatch preserved
    assert "delete_bucket" in text
