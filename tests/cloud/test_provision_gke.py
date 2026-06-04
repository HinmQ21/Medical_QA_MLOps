import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/provision_gke.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode_and_dry_run():
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "--dry-run" in text


def test_dry_run_prints_autopilot_commands_without_calling_cloud():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    assert "gcloud services enable" in out.stdout
    assert "container.googleapis.com" in out.stdout
    assert "gcloud container clusters create-auto medical-qa --region asia-southeast1" in out.stdout
    assert "get-credentials medical-qa --region asia-southeast1" in out.stdout


def test_is_idempotent_skips_create_when_cluster_exists():
    # provision must be safe to re-run (demo-up re-invokes it; operators may
    # pre-provision to create the Workload Identity pool before binding).
    assert "clusters describe" in SCRIPT.read_text()
