from tests.deploy.helm_helpers import find_kind, render_chart


def test_servicemonitors_select_services_and_carry_release_label():
    resources = render_chart("monitoring")
    api_sm = find_kind(resources, "ServiceMonitor", "medical-qa-api")
    ret_sm = find_kind(resources, "ServiceMonitor", "medical-qa-retrieval")

    assert api_sm["metadata"]["labels"]["release"] == "monitoring"
    assert api_sm["spec"]["selector"]["matchLabels"]["app.kubernetes.io/name"] == "medical-qa-api"
    assert api_sm["spec"]["endpoints"][0]["port"] == "http"
    assert api_sm["spec"]["endpoints"][0]["path"] == "/metrics"

    assert ret_sm["spec"]["selector"]["matchLabels"]["app.kubernetes.io/name"] == "medical-qa-retrieval"
    assert ret_sm["spec"]["endpoints"][0]["path"] == "/metrics"
