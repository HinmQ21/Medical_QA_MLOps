import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/setup_github_oidc.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo", "GITHUB_REPO": "octo/medical"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode():
    assert "set -euo pipefail" in SCRIPT.read_text()


def test_dry_run_creates_pool_provider_gsa_and_repo_scoped_binding():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    assert "workload-identity-pools create github-pool" in o
    assert "workload-identity-pools providers create-oidc github-provider" in o
    assert "https://token.actions.githubusercontent.com" in o
    assert "attribute.repository=assertion.repository" in o
    assert "assertion.repository=='octo/medical'" in o          # provider attribute-condition
    assert "iam service-accounts create medical-qa-deployer" in o
    assert "roles/container.admin" in o
    assert "roles/serviceusage.serviceUsageAdmin" in o
    assert "roles/iam.workloadIdentityUser" in o
    assert "attribute.repository/octo/medical" in o             # principalSet binding


def test_dry_run_prints_github_vars_to_set():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    o = out.stdout
    assert "GCP_WIF_PROVIDER=" in o
    assert "GCP_DEPLOY_SA=medical-qa-deployer@demo.iam.gserviceaccount.com" in o
    assert "GCP_PROJECT=demo" in o
