from pathlib import Path

from tests.deploy.helm_helpers import find_kind, render_chart


ROOT = Path(__file__).parents[2]


def test_nginx_configmap_template_does_not_contain_literal_default_key():
    text = (ROOT / "deploy/helm/nginx/templates/configmap.yaml").read_text()
    assert "${API_KEY}" in text
    assert "change-me-dev-key" not in text
    assert "proxy_pass http://medical-qa-api:8000" in text


def test_nginx_chart_renders_configmap_secret_deployment_and_service():
    resources = render_chart("nginx")
    config = find_kind(resources, "ConfigMap", "medical-qa-nginx-config")
    secret = find_kind(resources, "Secret", "medical-qa-nginx-api-key")
    deployment = find_kind(resources, "Deployment", "medical-qa-nginx")
    service = find_kind(resources, "Service", "medical-qa-nginx")

    conf = config["data"]["default.conf.template"]
    assert "${API_KEY}" in conf
    assert "change-me-dev-key" not in conf
    assert "proxy_pass http://medical-qa-api:8000" in conf
    assert secret["stringData"]["API_KEY"] == "change-me-dev-key"
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    env = container["env"][0]
    assert env["name"] == "API_KEY"
    assert env["valueFrom"]["secretKeyRef"]["name"] == "medical-qa-nginx-api-key"
    assert service["spec"]["ports"][0]["port"] == 8080
