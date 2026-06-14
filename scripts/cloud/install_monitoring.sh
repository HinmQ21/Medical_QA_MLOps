#!/usr/bin/env bash
# Install kube-prometheus-stack (Prometheus Operator + Grafana + Alertmanager) onto the
# current-context cluster. Idempotent (helm upgrade --install). Provides the
# servicemonitors.monitoring.coreos.com CRD that deploy.sh gates the monitoring chart on.
# Grafana's dashboard sidecar is enabled so the monitoring chart's dashboard ConfigMap
# is auto-imported. Version is pinned (Helm will not auto-resolve "latest").
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cloud/config.sh
source "$HERE/config.sh"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
HELM="$REPO_ROOT/.tools/bin/helm"
KUBECTL="$REPO_ROOT/.tools/bin/kubectl"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

run() {
  if [ "$DRY_RUN" -eq 1 ]; then echo "+ $*"; else "$@"; fi
}

# When Grafana is reached through a reverse proxy on a non-localhost host (GCP Cloud Shell
# Web Preview), its CSRF check rejects the proxied Origin and dashboards blank with "origin
# not allowed". Setting GRAFANA_ROOT_HOST teaches Grafana to trust that host via root_url +
# csrf_trusted_origins. These land in the Helm release values, so they survive `helm upgrade`
# (unlike a manual `kubectl set env`). Empty (the default) = direct localhost access, no
# override. Grafana has no wildcard support, so the exact host is required.
GRAFANA_ENV_ARGS=()
if [ -n "${GRAFANA_ROOT_HOST:-}" ]; then
  GRAFANA_ENV_ARGS+=(
    --set-string "grafana.env.GF_SERVER_ROOT_URL=https://${GRAFANA_ROOT_HOST}/"
    --set-string "grafana.env.GF_SECURITY_CSRF_TRUSTED_ORIGINS=${GRAFANA_ROOT_HOST}"
  )
fi

run "$HELM" repo add prometheus-community https://prometheus-community.github.io/helm-charts
run "$HELM" repo update
run "$HELM" upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --version "$KUBE_PROM_STACK_VERSION" \
  --namespace monitoring --create-namespace \
  --set grafana.sidecar.dashboards.enabled=true \
  --set grafana.sidecar.dashboards.searchNamespace=ALL \
  --set alertmanager.enabled=true \
  ${GRAFANA_ENV_ARGS[@]+"${GRAFANA_ENV_ARGS[@]}"}
run "$KUBECTL" -n monitoring rollout status deploy/monitoring-kube-prometheus-operator --timeout=300s
echo "kube-prometheus-stack $KUBE_PROM_STACK_VERSION installed (Prometheus + Grafana + Alertmanager)."
