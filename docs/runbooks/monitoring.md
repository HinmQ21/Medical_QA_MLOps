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

### Accessing Grafana through a reverse proxy ("origin not allowed")

`http://localhost:3000` works with **zero config** — Grafana trusts `localhost`. Prefer this.

If instead you reach Grafana through a proxy on a **non-localhost host** — notably **GCP
Cloud Shell Web Preview** (`<port>-cs-<id>.cs-<region>.cloudshell.dev`) — Grafana 11's CSRF
check rejects the proxied `Origin` on `POST /api/ds/query` and every panel blanks with
**"origin not allowed"** (hover the panel's warning triangle to see it). The data pipeline is
fine; only the browser→Grafana hop is blocked. Note Prometheus's own UI has no such check, so
it works through the same proxy — use it to confirm the data is present (`mqa_build_info` etc).

Fix durably by telling Grafana to trust that host **at install time** (survives `helm
upgrade`, unlike a manual `kubectl set env`):

```bash
# bare host from your browser's URL bar — no scheme, no trailing slash:
GRAFANA_ROOT_HOST=3000-cs-<id>.cs-<region>.cloudshell.dev \
  bash scripts/cloud/install_monitoring.sh
```

Grafana has **no wildcard support** for `csrf_trusted_origins`, and the Cloud Shell host
changes per VM — so the exact host is required and must be re-set if it changes. See
`GRAFANA_ROOT_HOST` in `scripts/cloud/config.sh`.

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
