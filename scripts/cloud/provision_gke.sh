#!/usr/bin/env bash
# Provision a GKE Autopilot cluster for the Medical QA stack.
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
run gcloud container clusters create-auto "$GKE_CLUSTER" --region "$GCP_REGION"
run gcloud container clusters get-credentials "$GKE_CLUSTER" --region "$GCP_REGION"
echo "GKE Autopilot cluster '$GKE_CLUSTER' ready in $GCP_REGION."
