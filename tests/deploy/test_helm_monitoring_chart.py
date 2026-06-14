import json

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


def test_prometheusrule_defines_named_alerts():
    resources = render_chart("monitoring")
    rule = find_kind(resources, "PrometheusRule", "medical-qa-alerts")
    assert rule["metadata"]["labels"]["release"] == "monitoring"
    alert_names = {
        r["alert"]
        for group in rule["spec"]["groups"]
        for r in group["rules"]
    }
    assert alert_names == {
        "HighErrorRate",
        "HighLatencyP95",
        "RetrievalEmptyRateHigh",
        "TargetDown",
    }


def test_prometheusrule_thresholds_come_from_values():
    resources = render_chart("monitoring", set_values={"alerts.errorRateThreshold": "0.1"})
    rule = find_kind(resources, "PrometheusRule", "medical-qa-alerts")
    exprs = [
        r["expr"]
        for group in rule["spec"]["groups"]
        for r in group["rules"]
        if r["alert"] == "HighErrorRate"
    ]
    assert "0.1" in exprs[0]


def test_empty_rate_denominator_excludes_not_called():
    resources = render_chart("monitoring")
    rule = find_kind(resources, "PrometheusRule", "medical-qa-alerts")
    expr = next(
        r["expr"]
        for group in rule["spec"]["groups"]
        for r in group["rules"]
        if r["alert"] == "RetrievalEmptyRateHigh"
    )
    # numerator counts empty; denominator must restrict to actual tool calls.
    assert 'outcome="empty"' in expr
    assert 'outcome!="not_called"' in expr


def test_servicemonitors_pin_joblabel_for_targetdown():
    resources = render_chart("monitoring")
    for name in ("medical-qa-api", "medical-qa-retrieval"):
        sm = find_kind(resources, "ServiceMonitor", name)
        assert sm["spec"]["jobLabel"] == "app.kubernetes.io/name"


def test_dashboard_configmap_carries_valid_json_and_grafana_label():
    resources = render_chart("monitoring")
    cm = find_kind(resources, "ConfigMap", "medical-qa-dashboard")
    assert cm["metadata"]["labels"]["grafana_dashboard"] == "1"
    raw = cm["data"]["medical-qa-serving.json"]
    dashboard = json.loads(raw)  # must be valid JSON
    titles = {p["title"] for p in dashboard["panels"]}
    assert "Request rate by status" in titles
    assert "Latency p95 (total/model)" in titles
    assert dashboard["templating"]["list"][0]["query"] == "prometheus"
    for panel in dashboard["panels"]:
        assert panel["datasource"]["uid"] == "${datasource}"
        for target in panel["targets"]:
            assert target["datasource"]["uid"] == "${datasource}"
