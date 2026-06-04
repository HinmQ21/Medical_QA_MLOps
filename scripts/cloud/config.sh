# Shared configuration for Medical QA cloud-deploy scripts (slim Plan 4, GKE-only).
# Sourced by every scripts/cloud/*.sh. Override any value via the environment.
: "${GCP_PROJECT:?set GCP_PROJECT to your Google Cloud project id}"
: "${GCP_REGION:=asia-southeast1}"
: "${GKE_CLUSTER:=medical-qa}"
: "${K8S_NAMESPACE:=medical-qa}"
: "${DVC_BUCKET:=${GCP_PROJECT}-medical-qa-dvc}"
: "${GSA_NAME:=medical-qa-retrieval}"
: "${KSA_NAME:=medical-qa-retrieval}"
: "${GSA_EMAIL:=${GSA_NAME}@${GCP_PROJECT}.iam.gserviceaccount.com}"
: "${REGISTRY:=ghcr.io/hinmq21}"
: "${IMAGE_TAG:=latest}"
# GitHub OIDC / Workload Identity Federation (keyless GitHub Actions -> GCP)
: "${GITHUB_REPO:=HinmQ21/Medical_QA_MLOps}"
: "${WIF_POOL:=github-pool}"
: "${WIF_PROVIDER:=github-provider}"
: "${DEPLOY_GSA_NAME:=medical-qa-deployer}"
: "${DEPLOY_GSA_EMAIL:=${DEPLOY_GSA_NAME}@${GCP_PROJECT}.iam.gserviceaccount.com}"
