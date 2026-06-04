import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/deploy.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode():
    assert "set -euo pipefail" in SCRIPT.read_text()


def test_dry_run_upgrades_four_charts_mock_backend():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    assert "upgrade --install medical-qa-retrieval deploy/helm/retrieval" in o
    assert "serviceAccount.gcpServiceAccount=medical-qa-retrieval@demo.iam.gserviceaccount.com" in o
    assert "dvc.bucket=demo-medical-qa-dvc" in o
    assert "--set-file dvc.yaml=dvc.yaml" in o  # real pipeline metadata so initContainer dvc pull fetches the KG
    assert "--set-file dvc.lock=dvc.lock" in o
    assert "upgrade --install medical-qa-api deploy/helm/api" in o
    assert "upgrade --install medical-qa-nginx deploy/helm/nginx" in o
    assert "deploy/helm/nginx/values-prod.yaml" in o
    assert "upgrade --install medical-qa-kserve deploy/helm/kserve" in o
    assert "--create-namespace" in o


def test_kserve_install_is_guarded_on_crd_presence():
    # The mock-backend demo runs on a vanilla Autopilot cluster with no KServe CRDs;
    # a hard `helm install` of the kserve chart there aborts with "no matches for kind
    # InferenceService" and fails the whole deploy. The live install must be guarded on
    # the CRD being present and skip (non-fatal) otherwise.
    text = SCRIPT.read_text()
    assert "get crd inferenceservices.serving.kserve.io" in text


def test_dry_run_does_not_use_runpod_or_api_prod_overlay():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    o = out.stdout
    assert "deploy/helm/api/values-prod.yaml" not in o  # slim: api stays mock
    assert "runpod" not in o.lower()
