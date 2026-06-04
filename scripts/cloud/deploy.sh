#!/usr/bin/env bash
# helm upgrade --install the four app-serving charts. api stays on the mock
# backend (GKE-only demo, no RunPod). nginx uses the LoadBalancer prod overlay;
# retrieval gets the Workload-Identity SA + the real GCS bucket.
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

cd "$REPO_ROOT"

run "$HELM" upgrade --install medical-qa-retrieval deploy/helm/retrieval \
  --namespace "$K8S_NAMESPACE" --create-namespace \
  --set image.tag="$IMAGE_TAG" \
  --set initImage.tag="$IMAGE_TAG" \
  --set serviceAccount.gcpServiceAccount="$GSA_EMAIL" \
  --set dvc.bucket="$DVC_BUCKET" \
  --set-file dvc.yaml=dvc.yaml \
  --set-file dvc.lock=dvc.lock

run "$HELM" upgrade --install medical-qa-api deploy/helm/api \
  --namespace "$K8S_NAMESPACE" \
  --set image.tag="$IMAGE_TAG"

run "$HELM" upgrade --install medical-qa-nginx deploy/helm/nginx \
  --namespace "$K8S_NAMESPACE" \
  -f deploy/helm/nginx/values-prod.yaml

# KServe is optional. The mock-backend demo needs no InferenceService, and a vanilla
# Autopilot cluster ships without KServe CRDs — a hard install there aborts with
# "no matches for kind InferenceService" and fails the whole deploy. Install the chart
# only when the CRD is present; otherwise skip (non-fatal).
KSERVE_INSTALL=("$HELM" upgrade --install medical-qa-kserve deploy/helm/kserve \
  --namespace "$K8S_NAMESPACE" \
  --set image.tag="$IMAGE_TAG")
if [ "$DRY_RUN" -eq 1 ]; then
  echo "+ ${KSERVE_INSTALL[*]}"
elif "$KUBECTL" get crd inferenceservices.serving.kserve.io >/dev/null 2>&1; then
  "${KSERVE_INSTALL[@]}"
else
  echo "KServe CRD (inferenceservices.serving.kserve.io) not found; skipping kserve chart (mock-backend demo does not need it)."
fi

echo "Deployed all charts (api=mock) to namespace $K8S_NAMESPACE."
