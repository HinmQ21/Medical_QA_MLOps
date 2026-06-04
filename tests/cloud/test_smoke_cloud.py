import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/smoke_cloud.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo", "NGINX_API_KEY": "prod-key"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode_and_requires_api_key():
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "NGINX_API_KEY:?" in text


def test_dry_run_resolves_lb_ip_and_hits_health_version_and_predict():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    assert "get service medical-qa-nginx" in o
    assert "loadBalancer.ingress[0].ip" in o
    assert "/health" in o
    assert "x-api-key: prod-key" in o
    assert "/version" in o
    assert "/predict" in o
    assert '"question"' in o
    assert '"options"' in o
