import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/provision_gke_standard.sh"
# in-cluster topology: a Standard zonal cluster distinct from the Autopilot one
ENV = {
    "PATH": "/usr/bin:/bin",
    "GCP_PROJECT": "demo",
    "GKE_CLUSTER": "medical-qa-llm",
    "GKE_LOCATION": "asia-southeast1-a",
}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode_and_dry_run():
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "--dry-run" in text


def test_dry_run_prints_standard_zonal_create_without_calling_cloud():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    assert "gcloud services enable" in o
    assert "container.googleapis.com" in o
    assert "container clusters create medical-qa-llm --location asia-southeast1-a" in o
    assert "--num-nodes 1" in o
    assert "--machine-type e2-standard-8" in o
    assert "--workload-pool=demo.svc.id.goog" in o
    assert "get-credentials medical-qa-llm --location asia-southeast1-a" in o


def test_is_idempotent_skips_create_when_cluster_exists():
    assert "clusters describe" in SCRIPT.read_text()
