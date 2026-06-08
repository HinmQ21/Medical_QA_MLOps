import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/install_kserve.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode_and_dry_run():
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "--dry-run" in text


def test_dry_run_installs_cert_manager_then_kserve_in_order():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    cm = o.index("upgrade --install cert-manager jetstack/cert-manager")
    crd = o.index("oci://ghcr.io/kserve/charts/kserve-crd --version v0.18.0")
    ctrl = o.index("oci://ghcr.io/kserve/charts/kserve-resources --version v0.18.0")
    assert cm < crd < ctrl  # cert-manager, then CRDs, then controller
    assert "--version v1.20.2" in o          # cert-manager pinned
    assert "crds.enabled=true" in o
    assert "kserve.controller.deploymentMode=RawDeployment" in o
