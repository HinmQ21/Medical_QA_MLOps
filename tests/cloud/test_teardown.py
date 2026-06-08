import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/teardown.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode():
    assert "set -euo pipefail" in SCRIPT.read_text()


def test_dry_run_uninstalls_nginx_then_deletes_cluster_keeps_bucket():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    nginx_idx = o.index("uninstall medical-qa-nginx")
    delete_idx = o.index("clusters delete medical-qa --location asia-southeast1")
    assert nginx_idx < delete_idx  # release the LoadBalancer before deleting the cluster
    assert "--quiet" in o
    assert "storage rm" not in o  # bucket is durable; not deleted here


def test_dry_run_deletes_the_zonal_cluster_when_topology_is_in_cluster():
    env = {**ENV, "GKE_CLUSTER": "medical-qa-llm", "GKE_LOCATION": "asia-southeast1-a"}
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr
    assert "clusters delete medical-qa-llm --location asia-southeast1-a" in out.stdout
