import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/demo_up_llm.sh"
# create_secrets.sh + smoke_cloud.sh require NGINX_API_KEY; deploy.sh (llm backend)
# requires LLM_BASE_URL/LLM_MODEL, which demo_up_llm.sh sets itself.
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo", "NGINX_API_KEY": "test-key"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode_and_dry_run():
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "--dry-run" in text


def test_dry_run_chains_the_building_blocks_in_order():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    prov = o.index("==> provision Standard zonal cluster")
    kserve = o.index("==> install cert-manager + KServe")
    secrets = o.index("==> create gateway + LLM secrets")
    deploy = o.index("==> deploy charts (llm backend)")
    smoke = o.index("==> smoke test")
    assert prov < kserve < secrets < deploy < smoke
    # routes the api at the in-cluster predictor with the llm backend
    assert "env.modelBackend=llm" in o
    assert "medical-qa-kserve-predictor.medical-qa.svc.cluster.local/v1" in o
    # waits for the InferenceService before smoke
    assert "wait --for=condition=Ready isvc/medical-qa-kserve" in o
    # provisions a ZONAL cluster (not the regional default) — free-trial cost guard
    assert "--location asia-southeast1-a" in o
