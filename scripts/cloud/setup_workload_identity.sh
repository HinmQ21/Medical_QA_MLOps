#!/usr/bin/env bash
# Bootstrap (run once): bind the retrieval Kubernetes SA to a GCP SA with
# read-only GCS access, so the dvc-pull initContainer authenticates keylessly via
# Workload Identity. The binding references [namespace/ksa] by name, so it stays
# valid for every freshly-provisioned cluster. The retrieval Helm chart sets the
# matching KSA annotation at deploy time.
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

run gcloud iam service-accounts create "$GSA_NAME" \
  --project "$GCP_PROJECT" \
  --display-name "Medical QA retrieval (DVC pull)" || true

run gcloud storage buckets add-iam-policy-binding "gs://$DVC_BUCKET" \
  --member "serviceAccount:$GSA_EMAIL" \
  --role roles/storage.objectViewer

run gcloud iam service-accounts add-iam-policy-binding "$GSA_EMAIL" \
  --project "$GCP_PROJECT" \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:${GCP_PROJECT}.svc.id.goog[${K8S_NAMESPACE}/${KSA_NAME}]"

echo "Workload Identity bound: ${K8S_NAMESPACE}/${KSA_NAME} -> ${GSA_EMAIL}"
echo "deploy.sh passes --set serviceAccount.gcpServiceAccount=${GSA_EMAIL} to the retrieval chart."
