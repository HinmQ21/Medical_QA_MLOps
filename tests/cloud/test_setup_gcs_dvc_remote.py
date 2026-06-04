import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/setup_gcs_dvc_remote.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode():
    assert "set -euo pipefail" in SCRIPT.read_text()


def test_dry_run_creates_bucket_and_configures_dvc_remote():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    assert "gcloud storage buckets create gs://demo-medical-qa-dvc" in out.stdout
    assert "--uniform-bucket-level-access" in out.stdout
    assert "dvc remote add -d -f gcsremote gs://demo-medical-qa-dvc/dvc" in out.stdout
    assert "dvc push" in out.stdout
