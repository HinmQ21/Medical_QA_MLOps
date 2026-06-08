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

# api defaults to the mock backend (GKE-only demo). Set MODEL_BACKEND=llm plus
# LLM_BASE_URL / LLM_MODEL to point at an OpenAI /v1 server (self-hosted vLLM on the
# DGX-Spark via Cloudflare Tunnel, or the in-cluster llama.cpp InferenceService); the
# LLM_API_KEY secret is created separately. "vllm" is accepted as a back-compat alias.
API_SET=(--set image.tag="$IMAGE_TAG")
BACKEND="${MODEL_BACKEND:-mock}"
if [ "$BACKEND" = "llm" ] || [ "$BACKEND" = "vllm" ]; then
  : "${LLM_BASE_URL:?set LLM_BASE_URL (e.g. https://llm.example/v1) for the llm backend}"
  : "${LLM_MODEL:?set LLM_MODEL (the served model name) for the llm backend}"
  API_SET+=(--set env.modelBackend="$BACKEND" \
            --set env.llmBaseUrl="$LLM_BASE_URL" \
            --set env.llmModel="$LLM_MODEL")
fi

run "$HELM" upgrade --install medical-qa-api deploy/helm/api \
  --namespace "$K8S_NAMESPACE" \
  "${API_SET[@]}"

run "$HELM" upgrade --install medical-qa-nginx deploy/helm/nginx \
  --namespace "$K8S_NAMESPACE" \
  -f deploy/helm/nginx/values-prod.yaml

# Streamlit demo UI on its own LoadBalancer; reaches the api through the nginx
# gateway in-cluster, reusing the gateway API-key Secret (medical-qa-nginx-api-key).
run "$HELM" upgrade --install medical-qa-ui deploy/helm/ui \
  --namespace "$K8S_NAMESPACE" \
  --set image.tag="$IMAGE_TAG" \
  -f deploy/helm/ui/values-prod.yaml

# KServe is optional. The default demo (mock or llm backend) needs no
# InferenceService, and a vanilla Autopilot cluster ships without KServe CRDs — a hard
# install there aborts with "no matches for kind InferenceService" and fails the whole
# deploy. When the CRD IS present (GKE Standard zonal with KServe installed), this
# deploys the real Qwen2.5-1.5B llama.cpp InferenceService; the image tag is pinned in
# the chart (upstream ghcr.io/ggml-org/llama.cpp:server), so IMAGE_TAG does not apply.
# To route the API at it, deploy with MODEL_BACKEND=llm and LLM_BASE_URL pointing at
# the in-cluster predictor Service.
KSERVE_INSTALL=("$HELM" upgrade --install medical-qa-kserve deploy/helm/kserve \
  --namespace "$K8S_NAMESPACE")
if [ "$DRY_RUN" -eq 1 ]; then
  echo "+ ${KSERVE_INSTALL[*]}"
elif "$KUBECTL" get crd inferenceservices.serving.kserve.io >/dev/null 2>&1; then
  "${KSERVE_INSTALL[@]}"
else
  echo "KServe CRD (inferenceservices.serving.kserve.io) not found; skipping kserve chart (mock/llm demo does not need it)."
fi

echo "Deployed all charts (api=$BACKEND) to namespace $K8S_NAMESPACE."
