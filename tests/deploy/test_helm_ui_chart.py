import pytest
import yaml

from tests.deploy.helm_helpers import find_kind, render_chart


def test_ui_base_chart_renders_configmap_deployment_service_clusterip():
    resources = render_chart("ui")
    config = find_kind(resources, "ConfigMap", "medical-qa-ui-config")
    deployment = find_kind(resources, "Deployment", "medical-qa-ui")
    service = find_kind(resources, "Service", "medical-qa-ui")

    assert config["data"]["API_BASE_URL"] == "http://medical-qa-nginx:8080"
    assert service["spec"]["type"] == "ClusterIP"
    assert service["spec"]["ports"][0]["port"] == 8501

    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["ports"][0]["containerPort"] == 8501
    assert container["livenessProbe"]["httpGet"]["path"] == "/_stcore/health"
    assert container["readinessProbe"]["httpGet"]["path"] == "/_stcore/health"

    env_by_name = {e["name"]: e for e in container.get("env", [])}
    ref = env_by_name["API_KEY"]["valueFrom"]["secretKeyRef"]
    assert ref["name"] == "medical-qa-nginx-api-key"
    assert ref["key"] == "API_KEY"


def test_ui_prod_overlay_switches_service_to_loadbalancer():
    resources = render_chart("ui", values_files=["values-prod.yaml"])
    service = find_kind(resources, "Service", "medical-qa-ui")
    assert service["spec"]["type"] == "LoadBalancer"
    assert service["spec"]["ports"][0]["port"] == 8501


def test_ui_values_point_at_nginx_gateway_in_cluster():
    from pathlib import Path

    root = Path(__file__).parents[2]
    values = yaml.safe_load((root / "deploy/helm/ui/values.yaml").read_text())
    assert values["env"]["apiBaseUrl"] == "http://medical-qa-nginx:8080"
    assert values["auth"]["existingSecret"] == "medical-qa-nginx-api-key"

def test_ui_chart_never_creates_a_secret():
    # The UI reuses medical-qa-nginx-api-key; it must never template its own Secret.
    for values in (None, ["values-prod.yaml"]):
        resources = render_chart("ui", values_files=values)
        kinds = [r.get("kind") for r in resources]
        assert "Secret" not in kinds, f"UI chart must not create a Secret (got {kinds})"
