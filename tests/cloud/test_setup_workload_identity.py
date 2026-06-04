import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/setup_workload_identity.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode():
    assert "set -euo pipefail" in SCRIPT.read_text()


def test_dry_run_binds_keyless_gcs_access():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    assert "gcloud iam service-accounts create medical-qa-retrieval" in out.stdout
    assert "buckets add-iam-policy-binding gs://demo-medical-qa-dvc" in out.stdout
    assert "roles/storage.objectViewer" in out.stdout
    assert "roles/iam.workloadIdentityUser" in out.stdout
    assert "demo.svc.id.goog[medical-qa/medical-qa-retrieval]" in out.stdout
