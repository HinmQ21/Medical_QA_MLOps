#!/usr/bin/env bash
# Tear down the ephemeral GKE demo: uninstall nginx (release the LoadBalancer)
# then delete the Autopilot cluster. The GCS bucket, GSAs, and WIF are durable and
# are NOT removed here (delete the bucket separately if you want a full clean-up).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cloud/config.sh
source "$HERE/config.sh"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
HELM="$REPO_ROOT/.tools/bin/helm"

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

# Release the public LoadBalancer first so deleting the cluster leaves no orphan.
run "$HELM" uninstall medical-qa-nginx --namespace "$K8S_NAMESPACE" || true
run gcloud container clusters delete "$GKE_CLUSTER" --region "$GCP_REGION" --quiet
echo "Cluster $GKE_CLUSTER deleted; LoadBalancer released. Bucket gs://$DVC_BUCKET kept."
