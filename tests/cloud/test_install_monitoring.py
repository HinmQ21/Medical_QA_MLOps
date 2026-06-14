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
    # kube-prometheus-stack chart versions are unprefixed semver (no leading "v").
    assert "--version 65.5.1" in o           # pinned KUBE_PROM_STACK_VERSION
    assert "--version v65.5.1" not in o      # guard against the KServe-style v-prefix mistake
    assert "grafana.sidecar.dashboards.enabled=true" in o
    assert "grafana.sidecar.dashboards.searchNamespace=ALL" in o
    assert "alertmanager.enabled=true" in o
    assert "rollout status deploy/monitoring-kube-prometheus-operator" in o


def test_dry_run_omits_grafana_origin_flags_by_default():
    # Direct localhost port-forward access needs no CSRF/root_url override (Grafana trusts
    # localhost). Default install must NOT weaken CSRF.
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    assert "GF_SECURITY_CSRF_TRUSTED_ORIGINS" not in out.stdout
    assert "GF_SERVER_ROOT_URL" not in out.stdout


def test_dry_run_sets_grafana_origin_when_root_host_given():
    # Reverse-proxy access (e.g. GCP Cloud Shell Web Preview) needs Grafana to trust the
    # proxied origin, else POST /api/ds/query is rejected with "origin not allowed".
    host = "3000-cs-abc.cs-asia-east1.cloudshell.dev"
    env = {**ENV, "GRAFANA_ROOT_HOST": host}
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    assert f"grafana.env.GF_SERVER_ROOT_URL=https://{host}/" in o
    assert f"grafana.env.GF_SECURITY_CSRF_TRUSTED_ORIGINS={host}" in o
