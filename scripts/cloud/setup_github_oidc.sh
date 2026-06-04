#!/usr/bin/env bash
# Bootstrap (run once): set up keyless GitHub Actions -> GCP auth via Workload
# Identity Federation, plus a least-privilege deploy GSA the workflows impersonate.
# Prints the GitHub repo variables to configure afterwards. No SA keys are created.
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

# Project number is needed for the principalSet member string.
if [ "$DRY_RUN" -eq 1 ]; then
  echo "+ gcloud projects describe $GCP_PROJECT --format=value(projectNumber)"
  PROJECT_NUMBER="PROJECT_NUMBER"
else
  PROJECT_NUMBER="$(gcloud projects describe "$GCP_PROJECT" --format='value(projectNumber)')"
fi

# 1. Workload Identity Pool + OIDC provider scoped to this exact GitHub repo.
run gcloud iam workload-identity-pools create "$WIF_POOL" \
  --project "$GCP_PROJECT" \
  --location global \
  --display-name "GitHub Actions pool" || true

run gcloud iam workload-identity-pools providers create-oidc "$WIF_PROVIDER" \
  --project "$GCP_PROJECT" \
  --location global \
  --workload-identity-pool "$WIF_POOL" \
  --display-name "GitHub provider" \
  --issuer-uri "https://token.actions.githubusercontent.com" \
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition "assertion.repository=='${GITHUB_REPO}'" || true

# 2. Least-privilege deploy GSA (manages clusters + uses APIs; NOT storage/IAM admin).
run gcloud iam service-accounts create "$DEPLOY_GSA_NAME" \
  --project "$GCP_PROJECT" \
  --display-name "Medical QA GitHub deployer" || true

for role in roles/container.admin roles/serviceusage.serviceUsageAdmin; do
  run gcloud projects add-iam-policy-binding "$GCP_PROJECT" \
    --member "serviceAccount:$DEPLOY_GSA_EMAIL" \
    --role "$role"
done

# 3. Allow the repo's OIDC identities to impersonate the deploy GSA.
PRINCIPAL="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL}/attribute.repository/${GITHUB_REPO}"
run gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_GSA_EMAIL" \
  --project "$GCP_PROJECT" \
  --role roles/iam.workloadIdentityUser \
  --member "$PRINCIPAL"

PROVIDER_RESOURCE="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL}/providers/${WIF_PROVIDER}"
echo ""
echo "Set these GitHub repo Variables (Settings > Secrets and variables > Actions > Variables):"
echo "  GCP_WIF_PROVIDER=${PROVIDER_RESOURCE}"
echo "  GCP_DEPLOY_SA=${DEPLOY_GSA_EMAIL}"
echo "  GCP_PROJECT=${GCP_PROJECT}"
echo "And one repo Secret: NGINX_API_KEY=<choose-a-strong-key>"
