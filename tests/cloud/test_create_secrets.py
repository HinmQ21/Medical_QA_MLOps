import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/create_secrets.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo", "NGINX_API_KEY": "prod-key"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode_and_requires_api_key():
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "NGINX_API_KEY:?" in text


def test_no_runpod_references():
    assert "RUNPOD" not in SCRIPT.read_text()


def test_dry_run_applies_nginx_secret_idempotently():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    assert "kubectl create secret generic medical-qa-nginx-api-key" in o
    assert "--from-literal=API_KEY=prod-key" in o
    assert "kubectl apply -f -" in o


def test_dry_run_ensures_namespace_before_secret():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    o = out.stdout
    assert "kubectl create namespace medical-qa" in o
    # namespace must be ensured before the secret is created in it
    assert o.index("create namespace medical-qa") < o.index(
        "create secret generic medical-qa-nginx-api-key"
    )


def test_missing_api_key_fails_fast():
    env = {k: v for k, v in ENV.items() if k != "NGINX_API_KEY"}
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=env
    )
    assert out.returncode != 0


def test_dry_run_creates_llm_secret_when_key_set():
    env = {**ENV, "LLM_API_KEY": "dgx-key"}
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    assert "kubectl create secret generic medical-qa-llm-key" in o
    assert "--from-literal=API_KEY=dgx-key" in o


def test_dry_run_skips_llm_secret_when_key_unset():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    assert "medical-qa-llm-key" not in out.stdout
