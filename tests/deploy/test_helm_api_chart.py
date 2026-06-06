from pathlib import Path

import yaml

from tests.deploy.helm_helpers import find_kind, render_chart


ROOT = Path(__file__).parents[2]


def test_api_values_configure_retrieval_service_not_localhost():
    values = yaml.safe_load((ROOT / "deploy/helm/api/values.yaml").read_text())
    assert values["env"]["retrievalUrl"] == "http://medical-qa-retrieval:8001"
    assert "localhost" not in values["env"]["retrievalUrl"]


def test_api_chart_renders_deployment_service_configmap_and_hpa():
    resources = render_chart("api")
    config = find_kind(resources, "ConfigMap", "medical-qa-api-config")
    deployment = find_kind(resources, "Deployment", "medical-qa-api")
    service = find_kind(resources, "Service", "medical-qa-api")
    hpa = find_kind(resources, "HorizontalPodAutoscaler", "medical-qa-api")

    assert config["data"]["RETRIEVAL_URL"] == "http://medical-qa-retrieval:8001"
    assert config["data"]["MODEL_BACKEND"] == "mock"
    # the default drift_log.jsonl resolves under root-owned /app; the non-root container
    # can't write it, so /predict 500s. Point drift logging at a writable dir.
    assert config["data"]["DRIFT_LOG_PATH"] == "/tmp/drift_log.jsonl"
    assert config["data"]["LLM_BASE_URL"] == ""
    assert config["data"]["LLM_MODEL"] == ""
    assert config["data"]["MAX_TOKENS"] == "2048"
    assert "RUNPOD_BASE_URL" not in config["data"]
    assert "RUNPOD_MODEL" not in config["data"]
    assert service["spec"]["ports"][0]["port"] == 8000
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["ports"][0]["containerPort"] == 8000
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["readinessProbe"]["httpGet"]["path"] == "/ready"
    env_by_name = {e["name"]: e for e in container.get("env", [])}
    ref = env_by_name["LLM_API_KEY"]["valueFrom"]["secretKeyRef"]
    assert ref["name"] == "medical-qa-llm-key"
    assert ref["key"] == "API_KEY"
    assert ref["optional"] is True
    assert hpa["apiVersion"] == "autoscaling/v2"
    assert hpa["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"] == 70
