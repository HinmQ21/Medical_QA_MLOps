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

run "$HELM" repo add prometheus-community https://prometheus-community.github.io/helm-charts
run "$HELM" repo update
run "$HELM" upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --version "$KUBE_PROM_STACK_VERSION" \
  --namespace monitoring --create-namespace \
  --set grafana.sidecar.dashboards.enabled=true \
  --set grafana.sidecar.dashboards.searchNamespace=ALL \
  --set alertmanager.enabled=true
run "$KUBECTL" -n monitoring rollout status deploy/monitoring-kube-prometheus-operator --timeout=300s
echo "kube-prometheus-stack $KUBE_PROM_STACK_VERSION installed (Prometheus + Grafana + Alertmanager)."
