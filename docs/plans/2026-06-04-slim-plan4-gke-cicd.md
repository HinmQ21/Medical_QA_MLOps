# Slim Plan 4 — GKE-only Demo + Automated CI/CD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the app-serving stack to a live GKE Autopilot cluster on the mock backend (no RunPod), with one-click `demo-up`/`demo-down` GitHub workflows, keyless GitHub→GCP auth, and auto-deploy on push while the cluster is alive.

**Architecture:** Reuse the Plan 4 cloud-script/chart-overlay pattern but drop the RunPod path (api stays `MODEL_BACKEND=mock`). Add a GitHub OIDC → GCP Workload Identity Federation bootstrap script and three workflows: `demo-up.yml` (workflow_dispatch: provision + deploy + smoke), `deploy.yml` (workflow_run after CI: roll updates only when the cluster is reachable, skip-green otherwise), `demo-down.yml` (workflow_dispatch: uninstall nginx + delete cluster, keep the durable GCS bucket). Durable resources (bucket, GSAs, WIF) are created once by an operator bootstrap; the CI deploy identity has least privilege. All scripts are `--dry-run`-testable; all gates run with no cloud calls.

**Tech Stack:** Bash, gcloud, kubectl, Helm, DVC (`dvc[gs]`), GKE Autopilot, GitHub Actions, Workload Identity Federation, pytest, PyYAML.

Implements the approved spec at `docs/specs/2026-06-04-slim-plan4-gke-cicd-design.md`.

---

## Scope

In scope: cloud config + `dvc[gs]` + Make targets; cloud scripts `provision_gke.sh`, `setup_gcs_dvc_remote.sh`, `setup_workload_identity.sh`, `setup_github_oidc.sh` (new), `create_secrets.sh` (nginx-only), `deploy.sh` (mock, no api overlay), `smoke_cloud.sh`, `teardown.sh` (new); retrieval + nginx Helm prod overlays; three GitHub workflows; runbook + README; tests.

Out of scope: RunPod / real 3B serving (api prod overlay intentionally omitted); Plan 5 (observability, MLflow server, drift, HPA load demo, Ingress/HTTPS); always-on cluster / GitOps controller.

## File Structure

**Create:**
- `scripts/cloud/config.sh` — shared sourced config (region/cluster/bucket/registry + WIF vars; no RunPod).
- `scripts/cloud/provision_gke.sh`
- `scripts/cloud/setup_gcs_dvc_remote.sh`
- `scripts/cloud/setup_workload_identity.sh`
- `scripts/cloud/setup_github_oidc.sh`
- `scripts/cloud/create_secrets.sh`
- `scripts/cloud/deploy.sh`
- `scripts/cloud/smoke_cloud.sh`
- `scripts/cloud/teardown.sh`
- `deploy/helm/retrieval/templates/serviceaccount.yaml`
- `deploy/helm/nginx/values-prod.yaml`
- `.github/workflows/demo-up.yml`
- `.github/workflows/demo-down.yml`
- `docs/cloud-setup.md`
- `tests/cloud/__init__.py`
- `tests/cloud/test_cloud_config_and_make.py`
- `tests/cloud/test_provision_gke.py`
- `tests/cloud/test_setup_gcs_dvc_remote.py`
- `tests/cloud/test_setup_workload_identity.py`
- `tests/cloud/test_setup_github_oidc.py`
- `tests/cloud/test_create_secrets.py`
- `tests/cloud/test_deploy_sh.py`
- `tests/cloud/test_smoke_cloud.py`
- `tests/cloud/test_teardown.py`
- `tests/deploy/test_helm_retrieval_prod.py`
- `tests/deploy/test_helm_nginx_prod.py`
- `tests/deploy/test_cloud_workflows.py`
- `tests/test_readme_cloud_docs.py`

**Modify:**
- `pyproject.toml` — `pipeline` extra `dvc[gs]>=3.50`.
- `Makefile` — add `cloud-*` targets.
- `tests/deploy/helm_helpers.py` — `render_chart` accepts values files + `--set`.
- `deploy/helm/retrieval/values.yaml` — add `serviceAccount`; replace `dvc.config` with `dvc.bucket`.
- `deploy/helm/retrieval/templates/configmap.yaml` — build `dvc-config` from `dvc.bucket`.
- `deploy/helm/retrieval/templates/deployment.yaml` — set `serviceAccountName`.
- `deploy/helm/nginx/values.yaml` — add `auth.existingSecret`, `service.type`.
- `deploy/helm/nginx/templates/secret.yaml` — skip when `existingSecret` set.
- `deploy/helm/nginx/templates/deployment.yaml` — secret name from `existingSecret`.
- `deploy/helm/nginx/templates/service.yaml` — `type` from values.
- `.github/workflows/deploy.yml` — rewrite to `workflow_run` auto-deploy with skip-if-no-cluster.
- `tests/deploy/test_ci_workflows.py` — replace the stale manual-dispatch test.
- `README.md` — pointer to the runbook.

---

## Task 1: Cloud config, `dvc[gs]` dependency, and Make targets

**Files:**
- Create: `scripts/cloud/config.sh`, `tests/cloud/__init__.py`, `tests/cloud/test_cloud_config_and_make.py`
- Modify: `pyproject.toml`, `Makefile`

- [ ] **Step 1: Create `tests/cloud/__init__.py`** (empty file, so the test package imports cleanly)

```python
```

- [ ] **Step 2: Write the failing test** `tests/cloud/test_cloud_config_and_make.py`

```python
import subprocess
from pathlib import Path

import tomllib

ROOT = Path(__file__).parents[2]


def test_pipeline_extra_uses_dvc_gcs_plugin():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pipeline = "\n".join(data["project"]["optional-dependencies"]["pipeline"])
    assert "dvc[gs]" in pipeline


def test_makefile_exposes_cloud_targets():
    text = (ROOT / "Makefile").read_text()
    for target in [
        "cloud-provision:",
        "cloud-gcs-dvc:",
        "cloud-workload-identity:",
        "cloud-github-oidc:",
        "cloud-secrets:",
        "cloud-deploy:",
        "cloud-smoke:",
        "cloud-teardown:",
    ]:
        assert target in text


def test_config_sh_defaults_region_bucket_and_oidc():
    text = (ROOT / "scripts/cloud/config.sh").read_text()
    assert "GCP_REGION:=asia-southeast1" in text
    assert "DVC_BUCKET:=${GCP_PROJECT}-medical-qa-dvc" in text
    assert "GCP_PROJECT:?" in text
    assert "WIF_POOL:=github-pool" in text
    assert "DEPLOY_GSA_NAME:=medical-qa-deployer" in text
    assert "RUNPOD" not in text  # slim plan: no RunPod config


def test_config_sh_applies_defaults_when_project_set():
    script = ROOT / "scripts/cloud/config.sh"
    out = subprocess.run(
        ["bash", "-c", f'GCP_PROJECT=demo source "{script}" && '
         'echo "$GCP_REGION $DVC_BUCKET $GKE_CLUSTER $DEPLOY_GSA_EMAIL"'],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == (
        "asia-southeast1 demo-medical-qa-dvc medical-qa "
        "medical-qa-deployer@demo.iam.gserviceaccount.com"
    )
```

- [ ] **Step 3: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/cloud/test_cloud_config_and_make.py -v
```
Expected: FAIL — `pyproject.toml` still has `dvc>=3.50`, no `cloud-*` targets, no `config.sh`.

- [ ] **Step 4: Update `pyproject.toml`**

In `[project.optional-dependencies]`, change the `pipeline` extra's `"dvc>=3.50",` line to:

```toml
    "dvc[gs]>=3.50",
```

Leave the rest of the `pipeline` extra (mlflow, PyYAML) unchanged.

- [ ] **Step 5: Create `scripts/cloud/config.sh`**

```bash
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
```

- [ ] **Step 6: Add `cloud-*` targets to `Makefile`**

Add the eight targets to the `.PHONY` line (append them to the existing list), and append these targets at the end of the file:

```makefile
cloud-provision:
	bash scripts/cloud/provision_gke.sh

cloud-gcs-dvc:
	bash scripts/cloud/setup_gcs_dvc_remote.sh

cloud-workload-identity:
	bash scripts/cloud/setup_workload_identity.sh

cloud-github-oidc:
	bash scripts/cloud/setup_github_oidc.sh

cloud-secrets:
	bash scripts/cloud/create_secrets.sh

cloud-deploy:
	bash scripts/cloud/deploy.sh

cloud-smoke:
	bash scripts/cloud/smoke_cloud.sh

cloud-teardown:
	bash scripts/cloud/teardown.sh
```

The `.PHONY` line becomes (append the new targets):
```makefile
.PHONY: install install-pipeline install-deploy-tools test smoke-pipeline smoke-pipeline-local mlflow-register-dry-run register-model dvc-status helm-lint helm-template helm-dry-run docker-build full-pipeline full-pipeline-dry-run smoke-full cloud-provision cloud-gcs-dvc cloud-workload-identity cloud-github-oidc cloud-secrets cloud-deploy cloud-smoke cloud-teardown
```

- [ ] **Step 7: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/cloud/test_cloud_config_and_make.py -v
```
Expected: 4 passed.

- [ ] **Step 8: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add pyproject.toml Makefile scripts/cloud/config.sh tests/cloud/__init__.py tests/cloud/test_cloud_config_and_make.py
git commit -m "chore: cloud config (GKE-only + OIDC), dvc[gs], cloud make targets"
```

---

## Task 2: GKE provisioning script

**Files:**
- Create: `scripts/cloud/provision_gke.sh`, `tests/cloud/test_provision_gke.py`

- [ ] **Step 1: Write the failing test** `tests/cloud/test_provision_gke.py`

```python
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/provision_gke.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode_and_dry_run():
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "--dry-run" in text


def test_dry_run_prints_autopilot_commands_without_calling_cloud():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    assert "gcloud services enable" in out.stdout
    assert "container.googleapis.com" in out.stdout
    assert "gcloud container clusters create-auto medical-qa --region asia-southeast1" in out.stdout
    assert "get-credentials medical-qa --region asia-southeast1" in out.stdout
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/cloud/test_provision_gke.py -v
```
Expected: FAIL — script does not exist.

- [ ] **Step 3: Create `scripts/cloud/provision_gke.sh`**

```bash
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
```

- [ ] **Step 4: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/cloud/test_provision_gke.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add scripts/cloud/provision_gke.sh tests/cloud/test_provision_gke.py
git commit -m "feat: add GKE Autopilot provisioning script"
```

---

## Task 3: GCS DVC remote script (bootstrap)

**Files:**
- Create: `scripts/cloud/setup_gcs_dvc_remote.sh`, `tests/cloud/test_setup_gcs_dvc_remote.py`

- [ ] **Step 1: Write the failing test** `tests/cloud/test_setup_gcs_dvc_remote.py`

```python
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/setup_gcs_dvc_remote.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode():
    assert "set -euo pipefail" in SCRIPT.read_text()


def test_dry_run_creates_bucket_and_configures_dvc_remote():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    assert "gcloud storage buckets create gs://demo-medical-qa-dvc" in out.stdout
    assert "--uniform-bucket-level-access" in out.stdout
    assert "dvc remote add -d -f gcsremote gs://demo-medical-qa-dvc/dvc" in out.stdout
    assert "dvc push" in out.stdout
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/cloud/test_setup_gcs_dvc_remote.py -v
```
Expected: FAIL — script does not exist.

- [ ] **Step 3: Create `scripts/cloud/setup_gcs_dvc_remote.sh`**

```bash
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
```

- [ ] **Step 4: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/cloud/test_setup_gcs_dvc_remote.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add scripts/cloud/setup_gcs_dvc_remote.sh tests/cloud/test_setup_gcs_dvc_remote.py
git commit -m "feat: add GCS DVC remote bootstrap script"
```

---

## Task 4: Workload Identity script (bootstrap)

**Files:**
- Create: `scripts/cloud/setup_workload_identity.sh`, `tests/cloud/test_setup_workload_identity.py`

- [ ] **Step 1: Write the failing test** `tests/cloud/test_setup_workload_identity.py`

```python
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/setup_workload_identity.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode():
    assert "set -euo pipefail" in SCRIPT.read_text()


def test_dry_run_binds_keyless_gcs_access():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    assert "gcloud iam service-accounts create medical-qa-retrieval" in out.stdout
    assert "buckets add-iam-policy-binding gs://demo-medical-qa-dvc" in out.stdout
    assert "roles/storage.objectViewer" in out.stdout
    assert "roles/iam.workloadIdentityUser" in out.stdout
    assert "demo.svc.id.goog[medical-qa/medical-qa-retrieval]" in out.stdout
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/cloud/test_setup_workload_identity.py -v
```
Expected: FAIL — script does not exist.

- [ ] **Step 3: Create `scripts/cloud/setup_workload_identity.sh`**

```bash
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
  --display-name "Medical QA retrieval (DVC pull)"

run gcloud storage buckets add-iam-policy-binding "gs://$DVC_BUCKET" \
  --member "serviceAccount:$GSA_EMAIL" \
  --role roles/storage.objectViewer

run gcloud iam service-accounts add-iam-policy-binding "$GSA_EMAIL" \
  --project "$GCP_PROJECT" \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:${GCP_PROJECT}.svc.id.goog[${K8S_NAMESPACE}/${KSA_NAME}]"

echo "Workload Identity bound: ${K8S_NAMESPACE}/${KSA_NAME} -> ${GSA_EMAIL}"
echo "deploy.sh passes --set serviceAccount.gcpServiceAccount=${GSA_EMAIL} to the retrieval chart."
```

- [ ] **Step 4: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/cloud/test_setup_workload_identity.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add scripts/cloud/setup_workload_identity.sh tests/cloud/test_setup_workload_identity.py
git commit -m "feat: add Workload Identity bootstrap script for retrieval GCS pull"
```

---

## Task 5: GitHub OIDC / Workload Identity Federation script (bootstrap)

**Files:**
- Create: `scripts/cloud/setup_github_oidc.sh`, `tests/cloud/test_setup_github_oidc.py`

- [ ] **Step 1: Write the failing test** `tests/cloud/test_setup_github_oidc.py`

```python
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/setup_github_oidc.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo", "GITHUB_REPO": "octo/medical"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode():
    assert "set -euo pipefail" in SCRIPT.read_text()


def test_dry_run_creates_pool_provider_gsa_and_repo_scoped_binding():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    assert "workload-identity-pools create github-pool" in o
    assert "workload-identity-pools providers create-oidc github-provider" in o
    assert "https://token.actions.githubusercontent.com" in o
    assert "attribute.repository=assertion.repository" in o
    assert "assertion.repository=='octo/medical'" in o          # provider attribute-condition
    assert "iam service-accounts create medical-qa-deployer" in o
    assert "roles/container.admin" in o
    assert "roles/serviceusage.serviceUsageAdmin" in o
    assert "roles/iam.workloadIdentityUser" in o
    assert "attribute.repository/octo/medical" in o             # principalSet binding


def test_dry_run_prints_github_vars_to_set():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    o = out.stdout
    assert "GCP_WIF_PROVIDER=" in o
    assert "GCP_DEPLOY_SA=medical-qa-deployer@demo.iam.gserviceaccount.com" in o
    assert "GCP_PROJECT=demo" in o
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/cloud/test_setup_github_oidc.py -v
```
Expected: FAIL — script does not exist.

- [ ] **Step 3: Create `scripts/cloud/setup_github_oidc.sh`**

```bash
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
  --display-name "GitHub Actions pool"

run gcloud iam workload-identity-pools providers create-oidc "$WIF_PROVIDER" \
  --project "$GCP_PROJECT" \
  --location global \
  --workload-identity-pool "$WIF_POOL" \
  --display-name "GitHub provider" \
  --issuer-uri "https://token.actions.githubusercontent.com" \
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition "assertion.repository=='${GITHUB_REPO}'"

# 2. Least-privilege deploy GSA (manages clusters + uses APIs; NOT storage/IAM admin).
run gcloud iam service-accounts create "$DEPLOY_GSA_NAME" \
  --project "$GCP_PROJECT" \
  --display-name "Medical QA GitHub deployer"

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
```

- [ ] **Step 4: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/cloud/test_setup_github_oidc.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add scripts/cloud/setup_github_oidc.sh tests/cloud/test_setup_github_oidc.py
git commit -m "feat: add GitHub OIDC / WIF bootstrap script for keyless deploys"
```

---

## Task 6: Runtime secrets script (nginx only)

**Files:**
- Create: `scripts/cloud/create_secrets.sh`, `tests/cloud/test_create_secrets.py`

- [ ] **Step 1: Write the failing test** `tests/cloud/test_create_secrets.py`

```python
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/create_secrets.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo", "NGINX_API_KEY": "prod-key"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode_and_requires_api_key():
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "NGINX_API_KEY:?" in text


def test_no_runpod_references():
    assert "RUNPOD" not in SCRIPT.read_text()


def test_dry_run_applies_nginx_secret_idempotently():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    assert "kubectl create secret generic medical-qa-nginx-api-key" in o
    assert "--from-literal=API_KEY=prod-key" in o
    assert "kubectl apply -f -" in o


def test_missing_api_key_fails_fast():
    env = {k: v for k, v in ENV.items() if k != "NGINX_API_KEY"}
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=env
    )
    assert out.returncode != 0
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/cloud/test_create_secrets.py -v
```
Expected: FAIL — script does not exist.

- [ ] **Step 3: Create `scripts/cloud/create_secrets.sh`**

```bash
#!/usr/bin/env bash
# Create the nginx gateway API-key Secret in the cluster. The GKE-only demo uses
# the mock model backend, so there is no RunPod secret. Value comes from the
# environment and is never written to the repo.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cloud/config.sh
source "$HERE/config.sh"

: "${NGINX_API_KEY:?set NGINX_API_KEY to the gateway API key}"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# kubectl create ... --dry-run=client | kubectl apply -f -  makes the secret idempotent.
apply_secret() {
  local name="$1"; shift
  local cmd="kubectl create secret generic $name --namespace $K8S_NAMESPACE $* --dry-run=client -o yaml | kubectl apply -f -"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "+ $cmd"
  else
    kubectl create secret generic "$name" --namespace "$K8S_NAMESPACE" "$@" --dry-run=client -o yaml | kubectl apply -f -
  fi
}

apply_secret medical-qa-nginx-api-key \
  "--from-literal=API_KEY=$NGINX_API_KEY"

echo "Secret medical-qa-nginx-api-key applied in namespace $K8S_NAMESPACE."
```

- [ ] **Step 4: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/cloud/test_create_secrets.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add scripts/cloud/create_secrets.sh tests/cloud/test_create_secrets.py
git commit -m "feat: add nginx API-key secret script (GKE-only, no RunPod)"
```

---

## Task 7: Extend the Helm render helper for overlays

**Files:**
- Modify: `tests/deploy/helm_helpers.py`

- [ ] **Step 1: Replace `render_chart` in `tests/deploy/helm_helpers.py`**

Replace the existing `render_chart` function (lines 20-44) with this signature-compatible version (existing no-arg callers are unaffected); keep the `timeout=30` and the `CalledProcessError` handling:

```python
def render_chart(
    chart_name: str,
    values_files: list[str] | None = None,
    set_values: dict[str, str] | None = None,
) -> list[dict]:
    helm = require_helm()
    chart = ROOT / "deploy/helm" / chart_name
    cmd = [str(helm), "template", f"test-{chart_name}", str(chart)]
    for values_file in values_files or []:
        cmd += ["-f", str(chart / values_file)]
    for key, value in (set_values or {}).items():
        cmd += ["--set", f"{key}={value}"]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout if exc.stdout is not None else exc.output
        raise AssertionError(
            "helm template failed\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{stdout or ''}\n"
            f"stderr:\n{exc.stderr or ''}"
        ) from exc
    return [
        doc
        for doc in yaml.safe_load_all(result.stdout)
        if isinstance(doc, dict) and doc
    ]
```

- [ ] **Step 2: Run existing chart tests to confirm no regression**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && make install-deploy-tools >/dev/null 2>&1; .venv/bin/pytest tests/deploy -v
```
Expected: all existing deploy tests still pass (render tests run because helm is installed).

- [ ] **Step 3: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add tests/deploy/helm_helpers.py
git commit -m "test: render_chart supports values files and set overrides"
```

---

## Task 8: Retrieval chart — Workload Identity SA + parametrized bucket

**Files:**
- Create: `deploy/helm/retrieval/templates/serviceaccount.yaml`, `tests/deploy/test_helm_retrieval_prod.py`
- Modify: `deploy/helm/retrieval/values.yaml`, `deploy/helm/retrieval/templates/configmap.yaml`, `deploy/helm/retrieval/templates/deployment.yaml`

- [ ] **Step 1: Write the failing test** `tests/deploy/test_helm_retrieval_prod.py`

```python
from tests.deploy.helm_helpers import find_kind, render_chart


def test_retrieval_renders_workload_identity_sa_and_real_bucket():
    resources = render_chart(
        "retrieval",
        set_values={
            "serviceAccount.gcpServiceAccount": "medical-qa-retrieval@demo.iam.gserviceaccount.com",
            "dvc.bucket": "demo-medical-qa-dvc",
        },
    )
    sa = find_kind(resources, "ServiceAccount", "medical-qa-retrieval")
    assert (
        sa["metadata"]["annotations"]["iam.gke.io/gcp-service-account"]
        == "medical-qa-retrieval@demo.iam.gserviceaccount.com"
    )
    deployment = find_kind(resources, "Deployment", "medical-qa-retrieval")
    assert deployment["spec"]["template"]["spec"]["serviceAccountName"] == "medical-qa-retrieval"
    config = find_kind(resources, "ConfigMap", "medical-qa-retrieval-dvc")
    assert "gs://demo-medical-qa-dvc/dvc" in config["data"]["dvc-config"]


def test_retrieval_base_render_still_has_sa_without_annotation():
    resources = render_chart("retrieval")
    sa = find_kind(resources, "ServiceAccount", "medical-qa-retrieval")
    assert "annotations" not in sa.get("metadata", {})
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_helm_retrieval_prod.py -v
```
Expected: FAIL — no ServiceAccount template; `dvc.bucket` not used.

- [ ] **Step 3: Edit `deploy/helm/retrieval/values.yaml`**

Add a `serviceAccount` block immediately after the `service:` block (after line 14):

```yaml
serviceAccount:
  create: true
  name: medical-qa-retrieval
  gcpServiceAccount: ""
```

Then replace the entire `dvc:` block (lines 29-40) — drop the `config: |` body and replace it with a `bucket` key, keeping `yaml`, `lock`, `artifactsDvc` exactly as they are:

```yaml
dvc:
  bucket: medical-qa-dvc-REPLACE-ME
  yaml: |
    stages: {}
  lock: |
    schema: '2.0'
  artifactsDvc: |
    outs: []
```

- [ ] **Step 4: Edit `deploy/helm/retrieval/templates/configmap.yaml`**

Replace the `dvc-config: |-` block (lines 6-7, which used `{{ .Values.dvc.config | indent 4 }}`) with a literal block built from the bucket value. Leave `dvc-yaml`, `dvc-lock`, and `artifacts-dvc` unchanged:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: medical-qa-retrieval-dvc
data:
  dvc-config: |-
    [core]
        remote = gcsremote
    ['remote "gcsremote"']
        url = gs://{{ .Values.dvc.bucket }}/dvc
  dvc-yaml: |-
{{ .Values.dvc.yaml | indent 4 }}
  dvc-lock: |-
{{ .Values.dvc.lock | indent 4 }}
  artifacts-dvc: |-
{{ .Values.dvc.artifactsDvc | indent 4 }}
```

- [ ] **Step 5: Create `deploy/helm/retrieval/templates/serviceaccount.yaml`**

```yaml
{{- if .Values.serviceAccount.create }}
apiVersion: v1
kind: ServiceAccount
metadata:
  name: {{ .Values.serviceAccount.name }}
{{- if .Values.serviceAccount.gcpServiceAccount }}
  annotations:
    iam.gke.io/gcp-service-account: {{ .Values.serviceAccount.gcpServiceAccount | quote }}
{{- end }}
{{- end }}
```

- [ ] **Step 6: Edit `deploy/helm/retrieval/templates/deployment.yaml`**

Add `serviceAccountName` as the first line under the pod `spec:` (line 17, immediately before `initContainers:`):

```yaml
    spec:
      serviceAccountName: {{ .Values.serviceAccount.name }}
      initContainers:
```

- [ ] **Step 7: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_helm_retrieval_prod.py tests/deploy/test_helm_retrieval_chart.py -v
```
Expected: new tests pass and the existing retrieval chart test still passes.

- [ ] **Step 8: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add deploy/helm/retrieval tests/deploy/test_helm_retrieval_prod.py
git commit -m "feat: retrieval chart Workload Identity SA and parametrized GCS bucket"
```

---

## Task 9: NGINX chart — existingSecret + LoadBalancer overlay

**Files:**
- Create: `deploy/helm/nginx/values-prod.yaml`, `tests/deploy/test_helm_nginx_prod.py`
- Modify: `deploy/helm/nginx/values.yaml`, `deploy/helm/nginx/templates/secret.yaml`, `deploy/helm/nginx/templates/deployment.yaml`, `deploy/helm/nginx/templates/service.yaml`

- [ ] **Step 1: Write the failing test** `tests/deploy/test_helm_nginx_prod.py`

```python
import pytest

from tests.deploy.helm_helpers import find_kind, render_chart


def test_nginx_prod_uses_loadbalancer_and_existing_secret_no_dev_key():
    resources = render_chart("nginx", values_files=["values-prod.yaml"])
    service = find_kind(resources, "Service", "medical-qa-nginx")
    assert service["spec"]["type"] == "LoadBalancer"
    deployment = find_kind(resources, "Deployment", "medical-qa-nginx")
    env = deployment["spec"]["template"]["spec"]["containers"][0]["env"][0]
    assert env["valueFrom"]["secretKeyRef"]["name"] == "medical-qa-nginx-api-key"
    with pytest.raises(AssertionError):
        find_kind(resources, "Secret", "medical-qa-nginx-api-key")


def test_nginx_base_still_creates_dev_secret_and_clusterip():
    resources = render_chart("nginx")
    service = find_kind(resources, "Service", "medical-qa-nginx")
    assert service["spec"]["type"] == "ClusterIP"
    secret = find_kind(resources, "Secret", "medical-qa-nginx-api-key")
    assert secret["stringData"]["API_KEY"] == "change-me-dev-key"
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_helm_nginx_prod.py -v
```
Expected: FAIL — no `values-prod.yaml`, secret always created, no `service.type`.

- [ ] **Step 3: Edit `deploy/helm/nginx/values.yaml`**

Under `auth:` add `existingSecret`, and replace the `service:` block to add `type` (keep `port: 8080`):

```yaml
auth:
  apiKey: change-me-dev-key
  existingSecret: ""

service:
  type: ClusterIP
  port: 8080
```

- [ ] **Step 4: Edit `deploy/helm/nginx/templates/secret.yaml`**

Wrap the whole manifest so it is only created when no existing secret is supplied:

```yaml
{{- if not .Values.auth.existingSecret }}
apiVersion: v1
kind: Secret
metadata:
  name: medical-qa-nginx-api-key
type: Opaque
stringData:
  API_KEY: {{ .Values.auth.apiKey | quote }}
{{- end }}
```

- [ ] **Step 5: Edit `deploy/helm/nginx/templates/deployment.yaml`**

Change the `secretKeyRef.name` line (line 25) from the literal `medical-qa-nginx-api-key` to a value with a default:

```yaml
                  name: {{ .Values.auth.existingSecret | default "medical-qa-nginx-api-key" }}
```

- [ ] **Step 6: Edit `deploy/helm/nginx/templates/service.yaml`**

Change `type: ClusterIP` (line 6) to read from values:

```yaml
  type: {{ .Values.service.type }}
```

- [ ] **Step 7: Create `deploy/helm/nginx/values-prod.yaml`**

```yaml
auth:
  existingSecret: medical-qa-nginx-api-key

service:
  type: LoadBalancer
```

- [ ] **Step 8: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_helm_nginx_prod.py tests/deploy/test_helm_nginx_chart.py -v
```
Expected: new tests pass and the existing nginx chart test still passes.

- [ ] **Step 9: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add deploy/helm/nginx tests/deploy/test_helm_nginx_prod.py
git commit -m "feat: nginx chart existingSecret and LoadBalancer prod overlay"
```

---

## Task 10: Deploy script (mock backend, no api overlay)

**Files:**
- Create: `scripts/cloud/deploy.sh`, `tests/cloud/test_deploy_sh.py`

- [ ] **Step 1: Write the failing test** `tests/cloud/test_deploy_sh.py`

```python
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/deploy.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode():
    assert "set -euo pipefail" in SCRIPT.read_text()


def test_dry_run_upgrades_four_charts_mock_backend():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    assert "upgrade --install medical-qa-retrieval deploy/helm/retrieval" in o
    assert "serviceAccount.gcpServiceAccount=medical-qa-retrieval@demo.iam.gserviceaccount.com" in o
    assert "dvc.bucket=demo-medical-qa-dvc" in o
    assert "upgrade --install medical-qa-api deploy/helm/api" in o
    assert "upgrade --install medical-qa-nginx deploy/helm/nginx" in o
    assert "deploy/helm/nginx/values-prod.yaml" in o
    assert "upgrade --install medical-qa-kserve deploy/helm/kserve" in o
    assert "--create-namespace" in o


def test_dry_run_does_not_use_runpod_or_api_prod_overlay():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    o = out.stdout
    assert "deploy/helm/api/values-prod.yaml" not in o  # slim: api stays mock
    assert "runpod" not in o.lower()
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/cloud/test_deploy_sh.py -v
```
Expected: FAIL — script does not exist.

- [ ] **Step 3: Create `scripts/cloud/deploy.sh`**

```bash
#!/usr/bin/env bash
# helm upgrade --install the four app-serving charts. api stays on the mock
# backend (GKE-only demo, no RunPod). nginx uses the LoadBalancer prod overlay;
# retrieval gets the Workload-Identity SA + the real GCS bucket.
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

cd "$REPO_ROOT"

run "$HELM" upgrade --install medical-qa-retrieval deploy/helm/retrieval \
  --namespace "$K8S_NAMESPACE" --create-namespace \
  --set image.tag="$IMAGE_TAG" \
  --set initImage.tag="$IMAGE_TAG" \
  --set serviceAccount.gcpServiceAccount="$GSA_EMAIL" \
  --set dvc.bucket="$DVC_BUCKET"

run "$HELM" upgrade --install medical-qa-api deploy/helm/api \
  --namespace "$K8S_NAMESPACE" \
  --set image.tag="$IMAGE_TAG"

run "$HELM" upgrade --install medical-qa-nginx deploy/helm/nginx \
  --namespace "$K8S_NAMESPACE" \
  -f deploy/helm/nginx/values-prod.yaml

run "$HELM" upgrade --install medical-qa-kserve deploy/helm/kserve \
  --namespace "$K8S_NAMESPACE" \
  --set image.tag="$IMAGE_TAG"

echo "Deployed all charts (api=mock) to namespace $K8S_NAMESPACE."
```

- [ ] **Step 4: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/cloud/test_deploy_sh.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add scripts/cloud/deploy.sh tests/cloud/test_deploy_sh.py
git commit -m "feat: add helm deploy script (mock backend, nginx LoadBalancer)"
```

---

## Task 11: Post-deploy smoke script

**Files:**
- Create: `scripts/cloud/smoke_cloud.sh`, `tests/cloud/test_smoke_cloud.py`

- [ ] **Step 1: Write the failing test** `tests/cloud/test_smoke_cloud.py`

```python
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/smoke_cloud.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo", "NGINX_API_KEY": "prod-key"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode_and_requires_api_key():
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "NGINX_API_KEY:?" in text


def test_dry_run_resolves_lb_ip_and_hits_health_version_and_predict():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    assert "get service medical-qa-nginx" in o
    assert "loadBalancer.ingress[0].ip" in o
    assert "/health" in o
    assert "x-api-key: prod-key" in o
    assert "/version" in o
    assert "/predict" in o
    assert '"question"' in o
    assert '"options"' in o
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/cloud/test_smoke_cloud.py -v
```
Expected: FAIL — script does not exist.

- [ ] **Step 3: Create `scripts/cloud/smoke_cloud.sh`**

```bash
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

BASE="http://${IP}:8080"
PAYLOAD='{"question":"Which medication is first-line for type 2 diabetes?","options":{"A":"Metformin","B":"Amoxicillin"}}'

run curl -fsS "$BASE/health"
run curl -fsS -H "x-api-key: $NGINX_API_KEY" "$BASE/version"
run curl -fsS -H "x-api-key: $NGINX_API_KEY" -H "content-type: application/json" \
  -X POST "$BASE/predict" -d "$PAYLOAD"
echo "Smoke test issued against $BASE."
```

- [ ] **Step 4: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/cloud/test_smoke_cloud.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add scripts/cloud/smoke_cloud.sh tests/cloud/test_smoke_cloud.py
git commit -m "feat: add post-deploy cloud smoke script (health/version/predict)"
```

---

## Task 12: Teardown script

**Files:**
- Create: `scripts/cloud/teardown.sh`, `tests/cloud/test_teardown.py`

- [ ] **Step 1: Write the failing test** `tests/cloud/test_teardown.py`

```python
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/teardown.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode():
    assert "set -euo pipefail" in SCRIPT.read_text()


def test_dry_run_uninstalls_nginx_then_deletes_cluster_keeps_bucket():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    nginx_idx = o.index("uninstall medical-qa-nginx")
    delete_idx = o.index("clusters delete medical-qa --region asia-southeast1")
    assert nginx_idx < delete_idx  # release the LoadBalancer before deleting the cluster
    assert "--quiet" in o
    assert "storage rm" not in o  # bucket is durable; not deleted here
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/cloud/test_teardown.py -v
```
Expected: FAIL — script does not exist.

- [ ] **Step 3: Create `scripts/cloud/teardown.sh`**

```bash
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
```

- [ ] **Step 4: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/cloud/test_teardown.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add scripts/cloud/teardown.sh tests/cloud/test_teardown.py
git commit -m "feat: add ephemeral teardown script (release LB, delete cluster, keep bucket)"
```

---

## Task 13: `demo-up.yml` workflow

**Files:**
- Create: `.github/workflows/demo-up.yml`, `tests/deploy/test_cloud_workflows.py`

- [ ] **Step 1: Write the failing test** `tests/deploy/test_cloud_workflows.py`

```python
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
WF = ROOT / ".github/workflows"


def _load(name: str) -> dict:
    return yaml.safe_load((WF / name).read_text())


def _triggers(wf: dict):
    # PyYAML parses the bare key `on:` as the boolean True.
    return wf.get("on", wf.get(True))


def test_demo_up_is_dispatch_and_uses_oidc_and_scripts():
    text = (WF / "demo-up.yml").read_text()
    wf = _load("demo-up.yml")
    assert "workflow_dispatch" in _triggers(wf)
    assert "id-token: write" in text                       # OIDC token permission
    assert "google-github-actions/auth@v2" in text
    assert "workload_identity_provider: ${{ vars.GCP_WIF_PROVIDER }}" in text
    assert "service_account: ${{ vars.GCP_DEPLOY_SA }}" in text
    assert "google-github-actions/get-gke-credentials@v2" in text
    assert "scripts/cloud/provision_gke.sh" in text
    assert "scripts/cloud/create_secrets.sh" in text
    assert "scripts/cloud/deploy.sh" in text
    assert "scripts/cloud/smoke_cloud.sh" in text
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_cloud_workflows.py -v
```
Expected: FAIL — `demo-up.yml` does not exist.

- [ ] **Step 3: Create `.github/workflows/demo-up.yml`**

```yaml
name: Demo Up (GKE)

on:
  workflow_dispatch:
    inputs:
      image_tag:
        description: Image tag to deploy
        required: true
        default: latest
      namespace:
        description: Kubernetes namespace
        required: true
        default: medical-qa

permissions:
  contents: read
  id-token: write

jobs:
  demo-up:
    runs-on: ubuntu-latest
    environment: gke-demo
    env:
      GCP_PROJECT: ${{ vars.GCP_PROJECT }}
      IMAGE_TAG: ${{ inputs.image_tag }}
      K8S_NAMESPACE: ${{ inputs.namespace }}
      NGINX_API_KEY: ${{ secrets.NGINX_API_KEY }}
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install pipeline + deploy tools
        run: |
          make install-pipeline
          make install-deploy-tools

      - id: auth
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ vars.GCP_WIF_PROVIDER }}
          service_account: ${{ vars.GCP_DEPLOY_SA }}

      - uses: google-github-actions/setup-gcloud@v2

      - name: Provision GKE Autopilot
        run: bash scripts/cloud/provision_gke.sh

      - uses: google-github-actions/get-gke-credentials@v2
        with:
          cluster_name: medical-qa
          location: asia-southeast1

      - name: Create nginx API-key secret
        run: bash scripts/cloud/create_secrets.sh

      - name: Deploy charts (mock backend)
        run: bash scripts/cloud/deploy.sh

      - name: Wait for nginx then smoke-test
        run: |
          .tools/bin/kubectl -n "$K8S_NAMESPACE" rollout status deploy/medical-qa-nginx --timeout=300s || true
          bash scripts/cloud/smoke_cloud.sh

      - name: Print public endpoint
        run: |
          IP=$(.tools/bin/kubectl get svc medical-qa-nginx -n "$K8S_NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
          echo "Demo is up at http://$IP:8080  (send header x-api-key: <NGINX_API_KEY>)"
```

- [ ] **Step 4: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_cloud_workflows.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add .github/workflows/demo-up.yml tests/deploy/test_cloud_workflows.py
git commit -m "ci: add demo-up workflow (provision + deploy + smoke, keyless OIDC)"
```

---

## Task 14: Rewrite `deploy.yml` to auto-deploy while the cluster is alive

**Files:**
- Modify: `.github/workflows/deploy.yml`, `tests/deploy/test_ci_workflows.py`, `tests/deploy/test_cloud_workflows.py`

- [ ] **Step 1: Replace the stale test in `tests/deploy/test_ci_workflows.py`**

Replace the whole `test_deploy_workflow_is_manual_dispatch_skeleton` function (lines 36-46) with a test of the new auto-deploy trigger (the manual helm-upgrade assertions now live in `test_deploy_sh.py` and the demo-up test):

```python
def test_deploy_workflow_auto_runs_after_ci_and_guards_cluster():
    text = (ROOT / ".github/workflows/deploy.yml").read_text()
    workflow = _workflow("deploy.yml")
    triggers = workflow.get("on", workflow.get(True))
    assert "workflow_run" in triggers
    assert triggers["workflow_run"]["workflows"] == ["CI"]
    assert "main" in triggers["workflow_run"]["branches"]
    commands = _all_run_commands(workflow)
    assert "gcloud container clusters describe medical-qa" in commands
    assert "up=false" in commands  # skip-green path when the cluster is down
    assert "scripts/cloud/deploy.sh" in commands
```

- [ ] **Step 2: Add the deploy-auto assertion to `tests/deploy/test_cloud_workflows.py`**

Append this function to `tests/deploy/test_cloud_workflows.py`:

```python
def test_deploy_auto_uses_oidc_and_is_gated_on_ci_success():
    text = (WF / "deploy.yml").read_text()
    assert "id-token: write" in text
    assert "google-github-actions/auth@v2" in text
    assert "workflow_run.conclusion == 'success'" in text
    assert "steps.cluster.outputs.up == 'true'" in text
    assert "google-github-actions/get-gke-credentials@v2" in text
```

- [ ] **Step 3: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_ci_workflows.py tests/deploy/test_cloud_workflows.py -v
```
Expected: FAIL — current `deploy.yml` is `workflow_dispatch`, has no `workflow_run`/cluster guard/OIDC.

- [ ] **Step 4: Replace `.github/workflows/deploy.yml` entirely**

```yaml
name: Auto Deploy

# Runs after CI finishes on main. Deploys the freshly-built images ONLY when the
# demo cluster is currently up; otherwise it skips and stays green (no red runs
# from pushes made while the demo is torn down). Bring the cluster up with the
# "Demo Up (GKE)" workflow; tear it down with "Demo Down (GKE)".
on:
  workflow_run:
    workflows: [CI]
    types: [completed]
    branches: [main]

permissions:
  contents: read
  id-token: write

jobs:
  deploy:
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest
    environment: gke-demo
    env:
      GCP_PROJECT: ${{ vars.GCP_PROJECT }}
      IMAGE_TAG: latest
      K8S_NAMESPACE: medical-qa
      NGINX_API_KEY: ${{ secrets.NGINX_API_KEY }}
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install deploy tools
        run: make install-deploy-tools

      - id: auth
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ vars.GCP_WIF_PROVIDER }}
          service_account: ${{ vars.GCP_DEPLOY_SA }}

      - uses: google-github-actions/setup-gcloud@v2

      - name: Detect whether the demo cluster is up
        id: cluster
        run: |
          if gcloud container clusters describe medical-qa --region asia-southeast1 >/dev/null 2>&1; then
            echo "up=true" >> "$GITHUB_OUTPUT"
          else
            echo "Cluster 'medical-qa' not found — demo is down. Skipping deploy."
            echo "up=false" >> "$GITHUB_OUTPUT"
          fi

      - if: steps.cluster.outputs.up == 'true'
        uses: google-github-actions/get-gke-credentials@v2
        with:
          cluster_name: medical-qa
          location: asia-southeast1

      - name: Rolling deploy
        if: steps.cluster.outputs.up == 'true'
        run: bash scripts/cloud/deploy.sh

      - name: Smoke-test
        if: steps.cluster.outputs.up == 'true'
        run: bash scripts/cloud/smoke_cloud.sh
```

- [ ] **Step 5: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_ci_workflows.py tests/deploy/test_cloud_workflows.py -v
```
Expected: all pass (the unchanged `test_ci_workflow_runs_tests_helm_checks_and_builds_images` still passes).

- [ ] **Step 6: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add .github/workflows/deploy.yml tests/deploy/test_ci_workflows.py tests/deploy/test_cloud_workflows.py
git commit -m "ci: deploy.yml auto-deploys after CI while cluster is alive, skips green otherwise"
```

---

## Task 15: `demo-down.yml` workflow

**Files:**
- Create: `.github/workflows/demo-down.yml`
- Modify: `tests/deploy/test_cloud_workflows.py`

- [ ] **Step 1: Append the failing test to `tests/deploy/test_cloud_workflows.py`**

```python
def test_demo_down_is_dispatch_tears_down_and_optionally_deletes_bucket():
    text = (WF / "demo-down.yml").read_text()
    wf = _load("demo-down.yml")
    assert "workflow_dispatch" in _triggers(wf)
    assert "id-token: write" in text
    assert "google-github-actions/auth@v2" in text
    assert "scripts/cloud/teardown.sh" in text
    assert "delete_bucket" in text
    assert "inputs.delete_bucket == 'true'" in text
    assert "storage rm -r" in text
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_cloud_workflows.py::test_demo_down_is_dispatch_tears_down_and_optionally_deletes_bucket -v
```
Expected: FAIL — `demo-down.yml` does not exist.

- [ ] **Step 3: Create `.github/workflows/demo-down.yml`**

```yaml
name: Demo Down (GKE)

on:
  workflow_dispatch:
    inputs:
      delete_bucket:
        description: Also delete the GCS DVC bucket (durable KG storage)
        required: true
        default: "false"

permissions:
  contents: read
  id-token: write

jobs:
  demo-down:
    runs-on: ubuntu-latest
    environment: gke-demo
    env:
      GCP_PROJECT: ${{ vars.GCP_PROJECT }}
      K8S_NAMESPACE: medical-qa
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install deploy tools
        run: make install-deploy-tools

      - id: auth
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ vars.GCP_WIF_PROVIDER }}
          service_account: ${{ vars.GCP_DEPLOY_SA }}

      - uses: google-github-actions/setup-gcloud@v2

      - name: Get cluster credentials
        run: gcloud container clusters get-credentials medical-qa --region asia-southeast1

      - name: Release LoadBalancer and delete cluster
        run: bash scripts/cloud/teardown.sh

      - name: Optionally delete the DVC bucket
        if: ${{ inputs.delete_bucket == 'true' }}
        run: gcloud storage rm -r "gs://${GCP_PROJECT}-medical-qa-dvc"
```

- [ ] **Step 4: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/deploy/test_cloud_workflows.py -v
```
Expected: all 3 cloud-workflow tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add .github/workflows/demo-down.yml tests/deploy/test_cloud_workflows.py
git commit -m "ci: add demo-down workflow (teardown, keep bucket unless delete_bucket=true)"
```

---

## Task 16: Cloud setup runbook + README pointer

**Files:**
- Create: `docs/cloud-setup.md`, `tests/test_readme_cloud_docs.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing test** `tests/test_readme_cloud_docs.py`

```python
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_runbook_documents_bootstrap_demo_and_teardown():
    text = (ROOT / "docs/cloud-setup.md").read_text()
    for needle in [
        "scripts/cloud/setup_github_oidc.sh",
        "scripts/cloud/setup_gcs_dvc_remote.sh",
        "scripts/cloud/setup_workload_identity.sh",
        "GCP_WIF_PROVIDER",
        "GCP_DEPLOY_SA",
        "NGINX_API_KEY",
        "Demo Up (GKE)",
        "Demo Down (GKE)",
        "Auto Deploy",
        "asia-southeast1",
        "mock",
        "Teardown",
    ]:
        assert needle in text, needle


def test_readme_points_to_cloud_runbook():
    text = (ROOT / "README.md").read_text()
    assert "docs/cloud-setup.md" in text
    assert "GKE" in text
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/test_readme_cloud_docs.py -v
```
Expected: FAIL — runbook and README pointer do not exist.

- [ ] **Step 3: Create `docs/cloud-setup.md`**

````markdown
# Cloud Setup Runbook — GKE-only Demo + Automated CI/CD (slim Plan 4)

Deploy the Medical QA stack to a live GKE Autopilot cluster with a cheap,
ephemeral, automated workflow. The model is served by the **mock** backend (no
RunPod) — this demonstrates the full request plumbing, real KG retrieval
(MedEmbed-small), autoscaling, and keyless CI/CD, not real 3B-model answers.

All `scripts/cloud/*.sh` accept `--dry-run` to print commands without calling the
cloud.

## How it works

- **CI** (`.github/workflows/ci.yml`) builds + pushes images to GHCR on every push
  to `main` (unchanged).
- **Demo Up (GKE)** (`demo-up.yml`, manual): provisions GKE, deploys the four
  charts on the mock backend, smoke-tests, and prints the public LoadBalancer IP.
- **Auto Deploy** (`deploy.yml`): runs automatically after CI succeeds on `main`.
  If the demo cluster is up it does a rolling `helm upgrade`; if it's down it skips
  and stays green. So "push code while the demo is live → it auto-updates."
- **Demo Down (GKE)** (`demo-down.yml`, manual): uninstalls nginx (releases the
  LoadBalancer) and deletes the cluster, stopping Autopilot + LB billing. The GCS
  bucket / GSAs / WIF are kept so the next demo is fast.

## One-time bootstrap (operator, from Cloud Shell or a machine with project owner)

Export configuration once per shell (region defaults to `asia-southeast1`):

```bash
export GCP_PROJECT=<your-project-id>
export GITHUB_REPO=HinmQ21/Medical_QA_MLOps   # override if your fork differs
```

1. **Enable APIs + create the GCS DVC remote and push the KG.** Generate the small
   KG artifacts first so `dvc push` has something to upload:

   ```bash
   gcloud auth login
   gcloud config set project "$GCP_PROJECT"
   gcloud services enable container.googleapis.com storage.googleapis.com iamcredentials.googleapis.com
   make smoke-pipeline                       # or your KG build that populates artifacts/
   bash scripts/cloud/setup_gcs_dvc_remote.sh
   git add .dvc/config && git commit -m "chore: point DVC at GCS remote"
   ```

2. **Bind keyless GCS access for the retrieval pod (Workload Identity):**

   ```bash
   bash scripts/cloud/setup_workload_identity.sh
   ```

3. **Set up keyless GitHub Actions → GCP auth (Workload Identity Federation):**

   ```bash
   bash scripts/cloud/setup_github_oidc.sh
   ```

   It prints three values. In GitHub → **Settings → Secrets and variables →
   Actions → Variables**, add:
   - `GCP_WIF_PROVIDER` = the printed provider resource name
   - `GCP_DEPLOY_SA` = the printed deploy service-account email
   - `GCP_PROJECT` = your project id

   And add one repo **Secret**:
   - `NGINX_API_KEY` = a strong key (the gateway `x-api-key`)

   Optionally create a GitHub **Environment** named `gke-demo` for approvals.

## Running the demo (no terminal needed)

1. GitHub → **Actions → Demo Up (GKE) → Run workflow**. Wait for the run; the last
   step prints `http://<LB-IP>:8080`.
2. Call it (mock answer):

   ```bash
   curl -fsS -H "x-api-key: <NGINX_API_KEY>" -H "content-type: application/json" \
     -X POST "http://<LB-IP>:8080/predict" \
     -d '{"question":"Which drug is first-line for type 2 diabetes?","options":{"A":"Metformin","B":"Amoxicillin"}}'
   curl -fsS -H "x-api-key: <NGINX_API_KEY>" "http://<LB-IP>:8080/version"
   ```
3. While the cluster is up, push to `main` → **Auto Deploy** rolls the new images
   automatically.
4. When done: GitHub → **Actions → Demo Down (GKE) → Run workflow** (leave
   `delete_bucket=false` to keep the KG for next time).

## Teardown (cost control)

`Demo Down (GKE)` runs `scripts/cloud/teardown.sh`, which uninstalls nginx (frees
the LoadBalancer) then deletes the Autopilot cluster. To also remove durable
storage, run it with `delete_bucket=true`, or manually:

```bash
gcloud storage rm -r "gs://${GCP_PROJECT}-medical-qa-dvc"
```

## Notes

- **Cost:** Autopilot pods + one public LoadBalancer accrue charges only while the
  cluster is up. An orphaned LoadBalancer keeps billing — always tear down via
  `Demo Down (GKE)` (it releases the LB before deleting the cluster).
- **Mock answers:** `/predict` returns a deterministic letter from the mock
  backend; retrieval is real.
- **RunPod (real 3B):** a future flip — set the api `modelBackend: runpod` + a
  RunPod secret; see the full Plan 4 spec.
````

- [ ] **Step 4: Edit `README.md`**

Append:

````markdown
## Cloud Deploy — GKE-only Demo (slim Plan 4)

Deploy the stack to GKE Autopilot with one-click GitHub workflows and keyless
CI/CD. The model uses the **mock** backend (no RunPod); retrieval is real. See the
full runbook in [`docs/cloud-setup.md`](docs/cloud-setup.md):

- **Demo Up (GKE)** / **Demo Down (GKE)** — manual workflows to bring the demo up
  (provision + deploy + smoke) and tear it down (release LB + delete cluster).
- **Auto Deploy** — pushes to `main` auto-roll new images while the cluster is up,
  and skip green when it's down.

One-time bootstrap (`setup_gcs_dvc_remote.sh`, `setup_workload_identity.sh`,
`setup_github_oidc.sh`) sets up the GCS DVC remote and keyless GitHub→GCP auth.
Live GKE deploy in `asia-southeast1`.
````

- [ ] **Step 5: Run to verify pass**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest tests/test_readme_cloud_docs.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add docs/cloud-setup.md README.md tests/test_readme_cloud_docs.py
git commit -m "docs: add GKE-only demo runbook and README pointer"
```

---

## Task 17: Full verification

**Files:**
- No new files expected.

- [ ] **Step 1: Install tools**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && make install-pipeline && make install-deploy-tools
```
Expected: `dvc[gs]` installed; `.tools/bin/helm` + `.tools/bin/kubectl` present.

- [ ] **Step 2: Run the full test suite with coverage**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && .venv/bin/pytest --cov=medical_qa_platform --cov-report=term-missing --cov-fail-under=80
```
Expected: all tests pass; coverage ≥ 80%. (The cloud scripts are bash, not Python, so they don't affect the Python coverage number.)

- [ ] **Step 3: Lint and render all charts (incl. overlays)**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && make helm-lint && make helm-template
.tools/bin/helm template x deploy/helm/nginx -f deploy/helm/nginx/values-prod.yaml >/dev/null
.tools/bin/helm template x deploy/helm/retrieval --set serviceAccount.gcpServiceAccount=a@b.iam.gserviceaccount.com --set dvc.bucket=demo-medical-qa-dvc >/dev/null
```
Expected: all charts lint and render cleanly with overlays.

- [ ] **Step 4: Dry-run every cloud script**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform
export GCP_PROJECT=demo NGINX_API_KEY=k GITHUB_REPO=octo/medical
for s in provision_gke setup_gcs_dvc_remote setup_workload_identity setup_github_oidc create_secrets deploy smoke_cloud teardown; do
  bash scripts/cloud/$s.sh --dry-run >/dev/null && echo "$s OK"
done
```
Expected: all eight print `OK` (no cloud calls).

- [ ] **Step 5: Validate every workflow YAML parses**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform
for w in .github/workflows/*.yml; do .venv/bin/python -c "import sys,yaml; yaml.safe_load(open('$w')); print('$w OK')"; done
```
Expected: `ci.yml`, `deploy.yml`, `demo-up.yml`, `demo-down.yml` all print `OK`.

- [ ] **Step 6: Inspect git status**

Run:
```bash
cd /home/vcsai/minhlbq/mlops-platform && git status --short
```
Expected: clean working tree after the task commits.

- [ ] **Step 7: Commit any verification-only fixes**

```bash
cd /home/vcsai/minhlbq/mlops-platform
git add pyproject.toml Makefile scripts deploy docs tests README.md .github
git commit -m "test: verify GKE-only demo + CI/CD toolchain" || true
```
Expected: a commit only if verification revealed necessary changes.

---

## Self-Review Notes

- **Spec coverage:** every spec deliverable maps to a task — config + `dvc[gs]` +
  Make targets (T1); provision (T2); GCS DVC remote (T3); retrieval Workload
  Identity (T4, T8); GitHub OIDC/WIF (T5); nginx-only secrets (T6); render helper
  (T7); nginx LoadBalancer/existingSecret overlay (T9); deploy on mock backend
  (T10); smoke incl. `/version` (T11); teardown keeping the bucket (T12); the three
  workflows (T13–T15) with the auto-deploy-while-alive skip-green logic; runbook +
  README (T16); verification (T17).
- **Slim vs full Plan 4:** no RunPod secret (T6 asserts `RUNPOD` absent), no api
  prod overlay (T10 asserts `api/values-prod.yaml` and `runpod` absent), api base
  stays `MODEL_BACKEND=mock` (existing api chart test unchanged).
- **Committed defaults stay safe:** base `values.yaml` keep ClusterIP /
  `change-me-dev-key` / no SA annotation; prod behavior is opt-in via
  `values-prod.yaml`/`--set` (asserted by base-render tests in T8–T9), so existing
  Plan 3 chart tests stay green.
- **No live credentials:** every script runs under `--dry-run` + `bash -n`; the
  nginx key comes from a GitHub Secret; GCP auth is keyless OIDC; the deploy GSA is
  least-privilege (`container.admin` + `serviceUsageAdmin` only — no storage/IAM
  admin, since bucket/GSA/WIF are created once at bootstrap).
- **Name consistency:** `medical-qa-nginx-api-key` (secret), `medical-qa-retrieval`
  (KSA=GSA name), `medical-qa-deployer` (deploy GSA), `github-pool`/`github-provider`
  (WIF), and repo vars `GCP_WIF_PROVIDER`/`GCP_DEPLOY_SA`/`GCP_PROJECT` match across
  `config.sh`, the scripts, the charts, and the three workflows.
- **YAML `on:` gotcha:** workflow tests read the `True` key (PyYAML parses bare
  `on:` as boolean) via the `_triggers` helper, matching the existing
  `test_ci_workflows.py` pattern.
- **Repository boundary:** no task imports from `baseline/`; all work stays inside
  `mlops-platform/`.
