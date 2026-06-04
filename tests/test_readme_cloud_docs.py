from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_runbook_documents_bootstrap_demo_and_teardown():
    text = (ROOT / "docs/cloud-setup.md").read_text()
    for needle in [
        "scripts/cloud/setup_github_oidc.sh",
        "scripts/cloud/setup_gcs_dvc_remote.sh",
        "scripts/cloud/setup_workload_identity.sh",
        "GCP_WIF_PROVIDER",
        "GCP_DEPLOY_SA",
        "NGINX_API_KEY",
        "Demo Up (GKE)",
        "Demo Down (GKE)",
        "Auto Deploy",
        "asia-southeast1",
        "mock",
        "Teardown",
    ]:
        assert needle in text, needle


def test_readme_points_to_cloud_runbook():
    text = (ROOT / "README.md").read_text()
    assert "docs/cloud-setup.md" in text
    assert "GKE" in text
