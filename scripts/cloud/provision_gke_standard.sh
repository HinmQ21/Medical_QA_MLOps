#!/usr/bin/env bash
# Provision a GKE Standard *zonal* cluster for the in-cluster LLM serving path.
# KServe RawDeployment needs the inferenceservices CRD, which Autopilot does not
# ship; this mirrors provision_gke.sh but uses `clusters create` (Standard) at one
# zone, with a single e2-standard-8 node (fits the 8-vCPU free-trial cap).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cloud/config.sh
source "$HERE/config.sh"

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

run gcloud config set project "$GCP_PROJECT"
run gcloud services enable \
  container.googleapis.com \
  storage.googleapis.com \
  iamcredentials.googleapis.com
# Idempotent: skip create if the cluster already exists, so demo-up can re-run safely.
if [ "$DRY_RUN" -eq 1 ]; then
  echo "+ gcloud container clusters describe $GKE_CLUSTER --location $GKE_LOCATION  # skip create if it exists"
  run gcloud container clusters create "$GKE_CLUSTER" --location "$GKE_LOCATION" \
    --num-nodes 1 --machine-type e2-standard-8 \
    --workload-pool="$GCP_PROJECT.svc.id.goog"
elif gcloud container clusters describe "$GKE_CLUSTER" --location "$GKE_LOCATION" >/dev/null 2>&1; then
  echo "Cluster $GKE_CLUSTER already exists in $GKE_LOCATION; skipping create."
else
  gcloud container clusters create "$GKE_CLUSTER" --location "$GKE_LOCATION" \
    --num-nodes 1 --machine-type e2-standard-8 \
    --workload-pool="$GCP_PROJECT.svc.id.goog"
fi
run gcloud container clusters get-credentials "$GKE_CLUSTER" --location "$GKE_LOCATION"
echo "GKE Standard cluster '$GKE_CLUSTER' ready in $GKE_LOCATION."
