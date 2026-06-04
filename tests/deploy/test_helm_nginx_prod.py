import pytest

from tests.deploy.helm_helpers import find_kind, render_chart


def test_nginx_prod_uses_loadbalancer_and_existing_secret_no_dev_key():
    resources = render_chart("nginx", values_files=["values-prod.yaml"])
    service = find_kind(resources, "Service", "medical-qa-nginx")
    assert service["spec"]["type"] == "LoadBalancer"
    deployment = find_kind(resources, "Deployment", "medical-qa-nginx")
    env = deployment["spec"]["template"]["spec"]["containers"][0]["env"][0]
    assert env["valueFrom"]["secretKeyRef"]["name"] == "medical-qa-nginx-api-key"
    with pytest.raises(AssertionError):
        find_kind(resources, "Secret", "medical-qa-nginx-api-key")


def test_nginx_base_still_creates_dev_secret_and_clusterip():
    resources = render_chart("nginx")
    service = find_kind(resources, "Service", "medical-qa-nginx")
    assert service["spec"]["type"] == "ClusterIP"
    secret = find_kind(resources, "Secret", "medical-qa-nginx-api-key")
    assert secret["stringData"]["API_KEY"] == "change-me-dev-key"
