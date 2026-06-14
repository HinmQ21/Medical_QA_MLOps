import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/install_monitoring.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode_and_dry_run():
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "--dry-run" in text


def test_dry_run_adds_repo_then_installs_stack_pinned():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    add = o.index("repo add prometheus-community")
    inst = o.index("upgrade --install monitoring prometheus-community/kube-prometheus-stack")
    assert add < inst
    assert "--version v65.5.1" in o          # pinned KUBE_PROM_STACK_VERSION
    assert "grafana.sidecar.dashboards.enabled=true" in o
    assert "grafana.sidecar.dashboards.searchNamespace=ALL" in o
