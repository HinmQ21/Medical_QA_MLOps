#!/usr/bin/env bash
# Bootstrap (run once): create the GCS bucket, configure it as the default DVC
# remote, and push the KG artifacts. The bucket is durable across demos.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cloud/config.sh
source "$HERE/config.sh"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

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

run gcloud storage buckets create "gs://$DVC_BUCKET" \
  --project "$GCP_PROJECT" \
  --location "$GCP_REGION" \
  --uniform-bucket-level-access

cd "$REPO_ROOT"
run .venv/bin/dvc remote add -d -f gcsremote "gs://$DVC_BUCKET/dvc"
run .venv/bin/dvc push
echo "DVC remote 'gcsremote' configured at gs://$DVC_BUCKET/dvc and artifacts pushed."
echo "Commit the updated .dvc/config to record the remote."
