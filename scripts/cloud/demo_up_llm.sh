#!/usr/bin/env bash
# One-command local bring-up of the in-cluster LLM demo (GKE Standard zonal +
# KServe RawDeployment + llama.cpp InferenceService), chaining the building-block
# scripts. Mirrors the demo-up.yml `llm_host: in-cluster` path for local use and for
# debugging a CI failure in Cloud Shell. Pass --dry-run to print the chain only.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
KUBECTL="$REPO_ROOT/.tools/bin/kubectl"

# In-cluster topology: a Standard zonal cluster distinct from the Autopilot one, with
# the api routed at the in-cluster predictor Service. Set before sourcing config so
# config defaults (GKE_ZONE etc.) are available, then derive the rest.
export GKE_CLUSTER="${GKE_CLUSTER:-medical-qa-llm}"
# Capture a caller-supplied location BEFORE config.sh defaults GKE_LOCATION to the
# region. This orchestrator always targets a *zonal* Standard cluster, so fall back
# to GKE_ZONE (not the region) when the caller didn't pin a location.
GKE_LOCATION_OVERRIDE="${GKE_LOCATION:-}"
# shellcheck source=scripts/cloud/config.sh
source "$HERE/config.sh"
export GKE_LOCATION="${GKE_LOCATION_OVERRIDE:-$GKE_ZONE}"
export MODEL_BACKEND=llm
export LLM_MODEL="${LLM_MODEL:-qwen2.5-1.5b-instruct}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://medical-qa-kserve-predictor.${K8S_NAMESPACE}.svc.cluster.local/v1}"

DRY_RUN=0
PASS=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1; PASS+=(--dry-run) ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

step() { echo "==> $1"; shift; "$@"; }

step "provision Standard zonal cluster" bash "$HERE/provision_gke_standard.sh" ${PASS[@]+"${PASS[@]}"}
step "install cert-manager + KServe"     bash "$HERE/install_kserve.sh" ${PASS[@]+"${PASS[@]}"}
step "create gateway + LLM secrets"      bash "$HERE/create_secrets.sh" ${PASS[@]+"${PASS[@]}"}
step "deploy charts (llm backend)"       bash "$HERE/deploy.sh" ${PASS[@]+"${PASS[@]}"}
if [ "$DRY_RUN" -eq 1 ]; then
  echo "+ $KUBECTL -n $K8S_NAMESPACE wait --for=condition=Ready isvc/medical-qa-kserve --timeout=600s"
else
  "$KUBECTL" -n "$K8S_NAMESPACE" wait --for=condition=Ready isvc/medical-qa-kserve --timeout=600s
fi
step "smoke test"                        bash "$HERE/smoke_cloud.sh" ${PASS[@]+"${PASS[@]}"}
echo "in-cluster LLM demo up."
