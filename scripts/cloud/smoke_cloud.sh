#!/usr/bin/env bash
# Smoke-test the live GKE deployment through the nginx LoadBalancer.
# api serves the mock backend, so /predict returns a deterministic answer letter.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cloud/config.sh
source "$HERE/config.sh"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
KUBECTL="$REPO_ROOT/.tools/bin/kubectl"

: "${NGINX_API_KEY:?set NGINX_API_KEY to the gateway API key}"

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

LB_QUERY='{.status.loadBalancer.ingress[0].ip}'
if [ "$DRY_RUN" -eq 1 ]; then
  echo "+ kubectl get service medical-qa-nginx --namespace $K8S_NAMESPACE -o jsonpath=$LB_QUERY"
  IP="LB_IP"
else
  IP="$("$KUBECTL" get service medical-qa-nginx --namespace "$K8S_NAMESPACE" -o "jsonpath=$LB_QUERY")"
fi

if [ "$DRY_RUN" -eq 0 ] && [ -z "$IP" ]; then
  echo "ERROR: nginx LoadBalancer has no external IP yet. Is the cluster ready? Wait ~30s and retry." >&2
  exit 1
fi

BASE="http://${IP}:8080"
PAYLOAD='{"question":"Which medication is first-line for type 2 diabetes?","options":{"A":"Metformin","B":"Amoxicillin"}}'

run curl -fsS "$BASE/health"
if [ "$DRY_RUN" -eq 1 ]; then
  echo "+ curl -fsS -H \"x-api-key: $NGINX_API_KEY\" $BASE/version"
  echo "+ curl -fsS -H \"x-api-key: $NGINX_API_KEY\" -H \"content-type: application/json\" -X POST $BASE/predict -d $PAYLOAD"
else
  curl -fsS -H "x-api-key: $NGINX_API_KEY" "$BASE/version" | grep -q '"contract_version"'
  curl -fsS -H "x-api-key: $NGINX_API_KEY" -H "content-type: application/json" \
    -X POST "$BASE/predict" -d "$PAYLOAD" | grep -q '"answer"'
fi
echo "Smoke test issued against $BASE."
