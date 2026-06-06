#!/usr/bin/env bash
# Create the nginx gateway API-key Secret in the cluster. The GKE-only demo uses
# the mock model backend, so there is no RunPod secret. Value comes from the
# environment and is never written to the repo.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cloud/config.sh
source "$HERE/config.sh"

: "${NGINX_API_KEY:?set NGINX_API_KEY to the gateway API key}"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# kubectl create ... --dry-run=client | kubectl apply -f -  makes the secret idempotent.
apply_secret() {
  local name="$1"; shift
  local cmd="kubectl create secret generic $name --namespace $K8S_NAMESPACE $* --dry-run=client -o yaml | kubectl apply -f -"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "+ $cmd"
  else
    kubectl create secret generic "$name" --namespace "$K8S_NAMESPACE" "$@" --dry-run=client -o yaml | kubectl apply -f -
  fi
}

# Ensure the target namespace exists before creating the secret in it (idempotent).
ensure_namespace() {
  local cmd="kubectl create namespace $K8S_NAMESPACE --dry-run=client -o yaml | kubectl apply -f -"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "+ $cmd"
  else
    kubectl create namespace "$K8S_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
  fi
}

ensure_namespace
apply_secret medical-qa-nginx-api-key \
  "--from-literal=API_KEY=$NGINX_API_KEY"

echo "Secret medical-qa-nginx-api-key applied in namespace $K8S_NAMESPACE."

# Optional LLM backend key (self-hosted vLLM on DGX-Spark via Cloudflare Tunnel).
# Created only when provided; the mock-backend demo leaves it unset and the api
# deployment treats the secret as optional.
if [ -n "${LLM_API_KEY:-}" ]; then
  apply_secret medical-qa-llm-key \
    "--from-literal=API_KEY=$LLM_API_KEY"
  echo "Secret medical-qa-llm-key applied in namespace $K8S_NAMESPACE."
else
  echo "LLM_API_KEY not set; skipping LLM secret (mock-backend demo)."
fi
