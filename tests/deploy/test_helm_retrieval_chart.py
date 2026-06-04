from pathlib import Path

import yaml

from tests.deploy.helm_helpers import find_kind, render_chart


ROOT = Path(__file__).parents[2]


def test_retrieval_values_use_runtime_and_dvc_init_images():
    values = yaml.safe_load((ROOT / "deploy/helm/retrieval/values.yaml").read_text())
    assert values["image"]["repository"].endswith("medical-qa-retrieval")
    assert values["initImage"]["repository"].endswith("medical-qa-pipeline-init")
    assert values["env"]["retrievalDevice"] == "cpu"
    assert values["env"]["kgEncoderModel"] == "abhinand/MedEmbed-small-v0.1"


def test_retrieval_chart_renders_dvc_init_container_pvc_and_hpa():
    resources = render_chart("retrieval")
    config = find_kind(resources, "ConfigMap", "medical-qa-retrieval-dvc")
    pvc = find_kind(resources, "PersistentVolumeClaim", "medical-qa-retrieval-artifacts")
    deployment = find_kind(resources, "Deployment", "medical-qa-retrieval")
    service = find_kind(resources, "Service", "medical-qa-retrieval")
    hpa = find_kind(resources, "HorizontalPodAutoscaler", "medical-qa-retrieval")

    assert {"dvc-config", "dvc-yaml", "dvc-lock", "artifacts-dvc"} <= set(config["data"])
    assert pvc["spec"]["resources"]["requests"]["storage"] == "5Gi"
    pod_spec = deployment["spec"]["template"]["spec"]
    init = pod_spec["initContainers"][0]
    container = pod_spec["containers"][0]
    assert init["name"] == "dvc-pull"
    assert init["image"].endswith("medical-qa-pipeline-init:latest")
    assert "dvc pull --no-run-cache" in init["args"][0]
    # scope the pull to the demo-KG stage: the serving pod needs the FAISS KG, not the
    # smoke eval/register artifacts — pulling everything couples startup to unrelated
    # outputs that may be absent from the remote and fails the whole init.
    assert "build_demo_kg" in init["args"][0]
    mount_names = {mount["name"] for mount in init["volumeMounts"]}
    assert {"artifacts", "dvc-metadata"} <= mount_names
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["RETRIEVAL_DEVICE"] == "cpu"
    assert env["KG_ENCODER_MODEL"] == "abhinand/MedEmbed-small-v0.1"
    assert env["KG_DATA_DIR"] == "/mnt/artifacts/demo/kg"
    assert env["HF_HOME"] == "/mnt/artifacts/hf"
    assert env["SENTENCE_TRANSFORMERS_HOME"] == "/mnt/artifacts/hf/sentence-transformers"
    assert service["spec"]["ports"][0]["port"] == 8001
    assert hpa["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"] == 70
