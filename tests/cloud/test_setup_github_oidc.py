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


def test_waits_for_deploy_sa_to_propagate_before_binding():
    # IAM is eventually consistent; the script must poll for the new SA before the
    # role bindings, or they intermittently fail with "Service account ... does not exist".
    text = SCRIPT.read_text()
    assert "service-accounts describe" in text
    sa_check = text.index("service-accounts describe")
    first_binding = text.index("add-iam-policy-binding")
    assert sa_check < first_binding  # the wait comes before any binding


def test_dry_run_grants_deploy_sa_act_as_default_compute_node_sa():
    # GKE create-auto provisions nodes that run as the default compute SA; without
    # iam.serviceAccountUser on it the deploy GSA gets HTTP 400 "does not have
    # access to service account ...-compute@developer..." and the cluster never builds.
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    o = out.stdout
    assert "roles/iam.serviceAccountUser" in o
    assert "compute@developer.gserviceaccount.com" in o
    # bound to the deploy GSA, not some other principal
    binding = next(ln for ln in o.splitlines() if "compute@developer.gserviceaccount.com" in ln)
    assert "roles/iam.serviceAccountUser" in binding
    assert "medical-qa-deployer@demo.iam.gserviceaccount.com" in binding


def test_dry_run_prints_github_vars_to_set():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    o = out.stdout
    assert "GCP_WIF_PROVIDER=" in o
    assert "GCP_DEPLOY_SA=medical-qa-deployer@demo.iam.gserviceaccount.com" in o
    assert "GCP_PROJECT=demo" in o
