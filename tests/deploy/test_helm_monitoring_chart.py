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
    assert ret_sm["metadata"]["labels"]["release"] == "monitoring"
    assert api_sm["spec"]["endpoints"][0]["interval"] == "30s"


def test_servicemonitor_selectors_match_real_service_labels():
    # A ServiceMonitor selects Services by their metadata.labels (NOT the pod selector).
    # Guard that the api/retrieval Services actually carry the labels the SMs select on,
    # otherwise Prometheus silently scrapes nothing.
    mon = render_chart("monitoring")
    for chart, svc_name in [("api", "medical-qa-api"), ("retrieval", "medical-qa-retrieval")]:
        svc = find_kind(render_chart(chart), "Service", svc_name)
        svc_labels = svc["metadata"].get("labels", {})
        sm = find_kind(mon, "ServiceMonitor", svc_name)
        selector = sm["spec"]["selector"]["matchLabels"]
        assert selector.items() <= svc_labels.items(), (
            f"{svc_name} ServiceMonitor selector {selector} not satisfied by "
            f"Service metadata.labels {svc_labels}"
        )
