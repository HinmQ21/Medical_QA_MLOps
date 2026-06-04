import subprocess
from pathlib import Path

import tomllib

ROOT = Path(__file__).parents[2]


def test_pipeline_extra_uses_dvc_gcs_plugin():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pipeline = "\n".join(data["project"]["optional-dependencies"]["pipeline"])
    assert "dvc[gs]" in pipeline


def test_makefile_exposes_cloud_targets():
    text = (ROOT / "Makefile").read_text()
    for target in [
        "cloud-provision:",
        "cloud-gcs-dvc:",
        "cloud-workload-identity:",
        "cloud-github-oidc:",
        "cloud-secrets:",
        "cloud-deploy:",
        "cloud-smoke:",
        "cloud-teardown:",
    ]:
        assert target in text


def test_config_sh_defaults_region_bucket_and_oidc():
    text = (ROOT / "scripts/cloud/config.sh").read_text()
    assert "GCP_REGION:=asia-southeast1" in text
    assert "DVC_BUCKET:=${GCP_PROJECT}-medical-qa-dvc" in text
    assert "GCP_PROJECT:?" in text
    assert "WIF_POOL:=github-pool" in text
    assert "DEPLOY_GSA_NAME:=medical-qa-deployer" in text
    assert "RUNPOD" not in text  # slim plan: no RunPod config


def test_config_sh_applies_defaults_when_project_set():
    script = ROOT / "scripts/cloud/config.sh"
    out = subprocess.run(
        ["bash", "-c", f'GCP_PROJECT=demo source "{script}" && '
         'echo "$GCP_REGION $DVC_BUCKET $GKE_CLUSTER $DEPLOY_GSA_EMAIL"'],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == (
        "asia-southeast1 demo-medical-qa-dvc medical-qa "
        "medical-qa-deployer@demo.iam.gserviceaccount.com"
    )
