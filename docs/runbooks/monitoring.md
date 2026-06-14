# Runbook: Monitoring (Prometheus + Grafana + Alertmanager)

Self-hosted `kube-prometheus-stack` scrapes the api/retrieval `/metrics`, renders the
"Medical QA Serving" Grafana dashboard, and evaluates the serving alerts.

## Install (once per cluster)

```bash
cd /home/vcsai/minhlbq/mlops-platform
bash scripts/cloud/install_monitoring.sh          # pinned KUBE_PROM_STACK_VERSION
# then (re)deploy so the monitoring chart's ServiceMonitors/rules/dashboard land:
IMAGE_TAG=<sha> bash scripts/cloud/deploy.sh
```

`deploy.sh` installs `deploy/helm/monitoring` only when the
`servicemonitors.monitoring.coreos.com` CRD exists — so run `install_monitoring.sh` first.

## View Grafana

```bash
.tools/bin/kubectl -n monitoring port-forward svc/monitoring-grafana 3000:80
# open http://localhost:3000 — default user admin
.tools/bin/kubectl -n monitoring get secret monitoring-grafana \
  -o jsonpath='{.data.admin-password}' | base64 -d; echo
```

The "Medical QA Serving" dashboard is auto-imported by the Grafana dashboard sidecar
(ConfigMap label `grafana_dashboard: "1"`).

## Verify scrape targets + alerts

```bash
.tools/bin/kubectl -n monitoring port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090
# http://localhost:9090/targets  -> medical-qa-api / medical-qa-retrieval should be "up"
# http://localhost:9090/alerts   -> HighErrorRate, HighLatencyP95, RetrievalEmptyRateHigh, TargetDown
```

## Notes

- Best validated on the GKE Standard in-cluster-LLM cluster (`medical-qa-llm`), where
  llama.cpp latency makes `mqa_model_latency_seconds` meaningful.
- Alert thresholds live in `deploy/helm/monitoring/values.yaml` (`alerts.*`).
- Alertmanager uses the stack default (no external receiver) — alerts are visible in the
  Alertmanager UI and Grafana, not routed to Slack/email.
- Pinned chart version is `KUBE_PROM_STACK_VERSION` in `scripts/cloud/config.sh` (unprefixed
  semver, e.g. `65.5.1`); confirm the version is reachable from `prometheus-community`
  before installing.
