#!/usr/bin/env bash
# Install cert-manager + KServe (RawDeployment) onto the current-context cluster.
# Idempotent (helm upgrade --install). Required before deploy.sh installs the kserve
# chart, which is gated on the inferenceservices.serving.kserve.io CRD being present.
# Gotchas baked in: KServe v0.17+ renamed the controller chart kserve->kserve-resources,
# and Helm 4 will not auto-resolve "latest" for OCI charts (so --version is explicit).
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

# cert-manager first — KServe's webhooks depend on it.
run "$HELM" repo add jetstack https://charts.jetstack.io
run "$HELM" repo update
run "$HELM" upgrade --install cert-manager jetstack/cert-manager \
  --namespace cert-manager --create-namespace \
  --version "$CERT_MANAGER_VERSION" --set crds.enabled=true
run "$KUBECTL" -n cert-manager rollout status deploy/cert-manager-webhook --timeout=300s

# KServe CRDs, then the controller in RawDeployment mode.
run "$HELM" upgrade --install kserve-crd \
  oci://ghcr.io/kserve/charts/kserve-crd --version "$KSERVE_VERSION"
run "$HELM" upgrade --install kserve \
  oci://ghcr.io/kserve/charts/kserve-resources --version "$KSERVE_VERSION" \
  --namespace kserve --create-namespace \
  --set kserve.controller.deploymentMode=RawDeployment
run "$KUBECTL" -n kserve rollout status deploy/kserve-controller-manager --timeout=300s
echo "cert-manager $CERT_MANAGER_VERSION and KServe $KSERVE_VERSION (RawDeployment) installed."
