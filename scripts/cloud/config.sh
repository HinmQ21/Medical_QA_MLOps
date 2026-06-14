# Shared configuration for Medical QA cloud-deploy scripts (slim Plan 4, GKE-only).
# Sourced by every scripts/cloud/*.sh. Override any value via the environment.
: "${GCP_PROJECT:?set GCP_PROJECT to your Google Cloud project id}"
: "${GCP_REGION:=asia-southeast1}"
: "${GKE_CLUSTER:=medical-qa}"
# Standard-zonal cluster location for the in-cluster LLM path (Autopilot uses the
# region). GKE_LOCATION defaults to the region so the existing regional flows are
# unchanged; the in-cluster path overrides it with the zone.
: "${GKE_ZONE:=${GCP_REGION}-a}"
: "${GKE_LOCATION:=$GCP_REGION}"
# Pinned versions for install_kserve.sh (cert-manager known-good from Path A;
# KServe controller chart is `kserve-resources` in v0.17+, installed via --version).
: "${CERT_MANAGER_VERSION:=v1.20.2}"
: "${KSERVE_VERSION:=v0.18.0}"
# Pinned kube-prometheus-stack chart version for install_monitoring.sh. NOTE: this
# Helm chart uses UNPREFIXED semver (65.5.1, not v65.5.1, unlike the KServe OCI charts).
: "${KUBE_PROM_STACK_VERSION:=65.5.1}"
# Optional Grafana external host for install_monitoring.sh. Grafana 11 rejects POST
# /api/ds/query when the browser Origin does not match its root_url ("origin not allowed"),
# which blanks dashboards when Grafana is reached through a reverse proxy with a non-localhost
# host — notably GCP Cloud Shell Web Preview ("<port>-cs-<id>.cs-<region>.cloudshell.dev").
# Set this to that bare host (no scheme/path) to configure Grafana's root_url + CSRF trusted
# origin. Leave empty for direct localhost port-forward access, which Grafana trusts by
# default (the robust, zero-config path). Grafana has NO wildcard support here and the Cloud
# Shell host changes per VM, so this is necessarily host-specific, not a static value.
: "${GRAFANA_ROOT_HOST:=}"
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
