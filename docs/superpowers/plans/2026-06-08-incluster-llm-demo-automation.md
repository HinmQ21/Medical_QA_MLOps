# In-cluster LLM Demo Automation ("Path B") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the in-cluster LLM demo (GKE Standard zonal + KServe RawDeployment + llama.cpp Qwen2.5-1.5B) reproducible without manual steps — via idempotent scripts plus a one-click `llm_host: in-cluster` path in the existing `demo-up`/`demo-down` workflows — and fix the nginx map_hash bug for real.

**Architecture:** "B on A" — the heavy logic lives in `scripts/cloud/*.sh` (runnable locally and in CI); the GitHub workflows only orchestrate and branch on a new `llm_host: dgx | in-cluster` input. The existing `dgx`/`mock` branches are wrapped under `if` with unchanged scripts; the only shared edits are additive config vars and one `--region`→`--location` swap. `deploy.sh`, `create_secrets.sh`, `smoke_cloud.sh` are already correct and untouched.

**Tech Stack:** Bash (`set -euo pipefail`, `--dry-run` convention), gcloud, Helm 4 (OCI charts), KServe `kserve-resources` v0.18.0 (RawDeployment), cert-manager v1.20.2, GitHub Actions (`workflow_dispatch`, keyless OIDC/WIF), pytest + PyYAML for workflow/chart/script tests.

---

## Spec

`docs/superpowers/specs/2026-06-08-incluster-llm-demo-automation-design.md`

## Operator prerequisites (verify once, not code)

Before running the `in-cluster` path, confirm the WIF deployer SA can create a Standard cluster and install cluster-scoped CRDs:

```bash
gcloud projects get-iam-policy "$GCP_PROJECT" \
  --flatten="bindings[].members" \
  --filter="bindings.members:medical-qa-deployer@${GCP_PROJECT}.iam.gserviceaccount.com" \
  --format="value(bindings.role)" | grep -q "roles/container.admin"
```

Expected: prints `roles/container.admin`. The identity that creates a GKE cluster automatically receives cluster-admin Kubernetes RBAC, which is what cert-manager/KServe need. If the binding is missing, grant it:
`gcloud projects add-iam-policy-binding "$GCP_PROJECT" --member="serviceAccount:medical-qa-deployer@${GCP_PROJECT}.iam.gserviceaccount.com" --role="roles/container.admin"`.

## Conventions for every task

- Run tests with the project venv: `.venv/bin/pytest <path> -v`.
- Bash scripts under `scripts/cloud/` follow the house style: `#!/usr/bin/env bash`, `set -euo pipefail`, a `--dry-run` flag whose `run()` echoes `+ <cmd>` instead of executing, and they `source "$HERE/config.sh"`.
- Make every shell script executable in the same commit: `chmod +x <script>` before `git add`.

---

### Task 1: Add zonal/location + version config vars to `config.sh`

**Files:**
- Modify: `scripts/cloud/config.sh`
- Test: `tests/cloud/test_cloud_config_and_make.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/cloud/test_cloud_config_and_make.py`:

```python
def test_config_sh_defines_zone_location_and_install_versions():
    script = ROOT / "scripts/cloud/config.sh"
    text = script.read_text()
    assert "GKE_ZONE:=${GCP_REGION}-a" in text
    assert "GKE_LOCATION:=$GCP_REGION" in text
    assert "CERT_MANAGER_VERSION:=v1.20.2" in text
    assert "KSERVE_VERSION:=v0.18.0" in text
    out = subprocess.run(
        ["bash", "-c", f'GCP_PROJECT=demo source "{script}" && '
         'echo "$GKE_ZONE $GKE_LOCATION $KSERVE_VERSION"'],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "asia-southeast1-a asia-southeast1 v0.18.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/cloud/test_cloud_config_and_make.py::test_config_sh_defines_zone_location_and_install_versions -v`
Expected: FAIL (`GKE_ZONE` unbound / assertion on stdout).

- [ ] **Step 3: Add the vars to `config.sh`**

In `scripts/cloud/config.sh`, immediately after the line `: "${GKE_CLUSTER:=medical-qa}"`, insert:

```bash
# Standard-zonal cluster location for the in-cluster LLM path (Autopilot uses the
# region). GKE_LOCATION defaults to the region so the existing regional flows are
# unchanged; the in-cluster path overrides it with the zone.
: "${GKE_ZONE:=${GCP_REGION}-a}"
: "${GKE_LOCATION:=$GCP_REGION}"
# Pinned versions for install_kserve.sh (cert-manager known-good from Path A;
# KServe controller chart is `kserve-resources` in v0.17+, installed via --version).
: "${CERT_MANAGER_VERSION:=v1.20.2}"
: "${KSERVE_VERSION:=v0.18.0}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/cloud/test_cloud_config_and_make.py -v`
Expected: PASS (all tests in the file, including the existing `test_config_sh_applies_defaults_when_project_set`, still pass — it only echoes the original four vars).

- [ ] **Step 5: Commit**

```bash
git add scripts/cloud/config.sh tests/cloud/test_cloud_config_and_make.py
git commit -m "feat(cloud): add GKE_ZONE/GKE_LOCATION + cert-manager/KServe version vars"
```

---

### Task 2: Make `teardown.sh` location-aware (`--region`→`--location`)

**Files:**
- Modify: `scripts/cloud/teardown.sh`
- Test: `tests/cloud/test_teardown.py`

- [ ] **Step 1: Update the existing test to expect `--location`**

In `tests/cloud/test_teardown.py`, replace the body of `test_dry_run_uninstalls_nginx_then_deletes_cluster_keeps_bucket` so the delete assertion uses `--location` (default `GKE_LOCATION` resolves to the region), and add a zonal case:

```python
def test_dry_run_uninstalls_nginx_then_deletes_cluster_keeps_bucket():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    nginx_idx = o.index("uninstall medical-qa-nginx")
    delete_idx = o.index("clusters delete medical-qa --location asia-southeast1")
    assert nginx_idx < delete_idx  # release the LoadBalancer before deleting the cluster
    assert "--quiet" in o
    assert "storage rm" not in o  # bucket is durable; not deleted here


def test_dry_run_deletes_the_zonal_cluster_when_topology_is_in_cluster():
    env = {**ENV, "GKE_CLUSTER": "medical-qa-llm", "GKE_LOCATION": "asia-southeast1-a"}
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr
    assert "clusters delete medical-qa-llm --location asia-southeast1-a" in out.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/cloud/test_teardown.py -v`
Expected: FAIL (script still emits `--region`).

- [ ] **Step 3: Change the delete command in `teardown.sh`**

In `scripts/cloud/teardown.sh`, replace:

```bash
run gcloud container clusters delete "$GKE_CLUSTER" --region "$GCP_REGION" --quiet
echo "Cluster $GKE_CLUSTER deleted; LoadBalancer released. Bucket gs://$DVC_BUCKET kept."
```

with:

```bash
# --location accepts a region (Autopilot) OR a zone (Standard in-cluster cluster).
# GKE_LOCATION defaults to the region, so the regional teardown is unchanged.
run gcloud container clusters delete "$GKE_CLUSTER" --location "$GKE_LOCATION" --quiet
echo "Cluster $GKE_CLUSTER deleted ($GKE_LOCATION); LoadBalancer released. Bucket gs://$DVC_BUCKET kept."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/cloud/test_teardown.py -v`
Expected: PASS (both regional and zonal cases).

- [ ] **Step 5: Commit**

```bash
git add scripts/cloud/teardown.sh tests/cloud/test_teardown.py
git commit -m "feat(cloud): teardown deletes by --location (supports zonal in-cluster cluster)"
```

---

### Task 3: New `provision_gke_standard.sh` (GKE Standard zonal)

**Files:**
- Create: `scripts/cloud/provision_gke_standard.sh`
- Test: `tests/cloud/test_provision_gke_standard.py`

- [ ] **Step 1: Write the failing test**

Create `tests/cloud/test_provision_gke_standard.py`:

```python
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/provision_gke_standard.sh"
# in-cluster topology: a Standard zonal cluster distinct from the Autopilot one
ENV = {
    "PATH": "/usr/bin:/bin",
    "GCP_PROJECT": "demo",
    "GKE_CLUSTER": "medical-qa-llm",
    "GKE_LOCATION": "asia-southeast1-a",
}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode_and_dry_run():
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "--dry-run" in text


def test_dry_run_prints_standard_zonal_create_without_calling_cloud():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    assert "gcloud services enable" in o
    assert "container.googleapis.com" in o
    assert "container clusters create medical-qa-llm --location asia-southeast1-a" in o
    assert "--num-nodes 1" in o
    assert "--machine-type e2-standard-8" in o
    assert "--workload-pool=demo.svc.id.goog" in o
    assert "get-credentials medical-qa-llm --location asia-southeast1-a" in o


def test_is_idempotent_skips_create_when_cluster_exists():
    assert "clusters describe" in SCRIPT.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/cloud/test_provision_gke_standard.py -v`
Expected: FAIL (script file does not exist).

- [ ] **Step 3: Create the script**

Create `scripts/cloud/provision_gke_standard.sh`:

```bash
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
```

- [ ] **Step 4: Make it executable and run the test**

Run:
```bash
chmod +x scripts/cloud/provision_gke_standard.sh
.venv/bin/pytest tests/cloud/test_provision_gke_standard.py -v
```
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/cloud/provision_gke_standard.sh tests/cloud/test_provision_gke_standard.py
git commit -m "feat(cloud): provision_gke_standard.sh (GKE Standard zonal for in-cluster LLM)"
```

---

### Task 4: New `install_kserve.sh` (cert-manager + KServe RawDeployment)

**Files:**
- Create: `scripts/cloud/install_kserve.sh`
- Test: `tests/cloud/test_install_kserve.py`

- [ ] **Step 1: Write the failing test**

Create `tests/cloud/test_install_kserve.py`:

```python
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/install_kserve.sh"
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode_and_dry_run():
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "--dry-run" in text


def test_dry_run_installs_cert_manager_then_kserve_in_order():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    cm = o.index("upgrade --install cert-manager jetstack/cert-manager")
    crd = o.index("oci://ghcr.io/kserve/charts/kserve-crd --version v0.18.0")
    ctrl = o.index("oci://ghcr.io/kserve/charts/kserve-resources --version v0.18.0")
    assert cm < crd < ctrl  # cert-manager, then CRDs, then controller
    assert "--version v1.20.2" in o          # cert-manager pinned
    assert "crds.enabled=true" in o
    assert "kserve.controller.deploymentMode=RawDeployment" in o
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/cloud/test_install_kserve.py -v`
Expected: FAIL (script file does not exist).

- [ ] **Step 3: Create the script**

Create `scripts/cloud/install_kserve.sh`:

```bash
#!/usr/bin/env bash
# Install cert-manager + KServe (RawDeployment) onto the current-context cluster.
# Idempotent (helm upgrade --install). Required before deploy.sh installs the kserve
# chart, which is gated on the inferenceservices.serving.kserve.io CRD being present.
# Gotchas baked in: KServe v0.17+ renamed the controller chart kserve->kserve-resources,
# and Helm 4 will not auto-resolve "latest" for OCI charts (so --version is explicit).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cloud/config.sh
source "$HERE/config.sh"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
HELM="$REPO_ROOT/.tools/bin/helm"
KUBECTL="$REPO_ROOT/.tools/bin/kubectl"

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

# cert-manager first — KServe's webhooks depend on it.
run "$HELM" repo add jetstack https://charts.jetstack.io
run "$HELM" repo update
run "$HELM" upgrade --install cert-manager jetstack/cert-manager \
  --namespace cert-manager --create-namespace \
  --version "$CERT_MANAGER_VERSION" --set crds.enabled=true
run "$KUBECTL" -n cert-manager rollout status deploy/cert-manager-webhook --timeout=300s

# KServe CRDs, then the controller in RawDeployment mode.
run "$HELM" upgrade --install kserve-crd \
  oci://ghcr.io/kserve/charts/kserve-crd --version "$KSERVE_VERSION"
run "$HELM" upgrade --install kserve \
  oci://ghcr.io/kserve/charts/kserve-resources --version "$KSERVE_VERSION" \
  --namespace kserve --create-namespace \
  --set kserve.controller.deploymentMode=RawDeployment
run "$KUBECTL" -n kserve rollout status deploy/kserve-controller-manager --timeout=300s
echo "cert-manager $CERT_MANAGER_VERSION and KServe $KSERVE_VERSION (RawDeployment) installed."
```

- [ ] **Step 4: Make it executable and run the test**

Run:
```bash
chmod +x scripts/cloud/install_kserve.sh
.venv/bin/pytest tests/cloud/test_install_kserve.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/cloud/install_kserve.sh tests/cloud/test_install_kserve.py
git commit -m "feat(cloud): install_kserve.sh (cert-manager + KServe RawDeployment, pinned)"
```

---

### Task 5: Fix the nginx map_hash bug in the chart

**Files:**
- Modify: `deploy/helm/nginx/templates/configmap.yaml`
- Test: `tests/deploy/test_helm_nginx_chart.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/deploy/test_helm_nginx_chart.py`:

```python
def test_nginx_configmap_sets_map_hash_bucket_size_for_long_keys():
    # A real API key (>=40 chars) overflows nginx's default map_hash_bucket_size 64
    # and crashloops with "could not build map_hash". 128 makes any key length safe.
    text = (ROOT / "deploy/helm/nginx/templates/configmap.yaml").read_text()
    assert "map_hash_bucket_size 128;" in text
    resources = render_chart("nginx")
    conf = find_kind(resources, "ConfigMap", "medical-qa-nginx-config")["data"]["default.conf.template"]
    assert "map_hash_bucket_size 128;" in conf
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/deploy/test_helm_nginx_chart.py::test_nginx_configmap_sets_map_hash_bucket_size_for_long_keys -v`
Expected: FAIL (`map_hash_bucket_size 128;` absent).

- [ ] **Step 3: Add the directive to the configmap**

In `deploy/helm/nginx/templates/configmap.yaml`, change:

```yaml
  default.conf.template: |-
    map $http_x_api_key $api_key_valid {
      default 0;
      "${API_KEY}" 1;
    }
```

to:

```yaml
  default.conf.template: |-
    map_hash_bucket_size 128;
    map $http_x_api_key $api_key_valid {
      default 0;
      "${API_KEY}" 1;
    }
```

- [ ] **Step 4: Run the full nginx chart test file to verify nothing else broke**

Run: `.venv/bin/pytest tests/deploy/test_helm_nginx_chart.py -v`
Expected: PASS (new test + the two existing tests).

- [ ] **Step 5: Commit**

```bash
git add deploy/helm/nginx/templates/configmap.yaml tests/deploy/test_helm_nginx_chart.py
git commit -m "fix(nginx): map_hash_bucket_size 128 so long API keys don't crashloop"
```

---

### Task 6: New `demo_up_llm.sh` local orchestrator

**Files:**
- Create: `scripts/cloud/demo_up_llm.sh`
- Test: `tests/cloud/test_demo_up_llm.py`

- [ ] **Step 1: Write the failing test**

Create `tests/cloud/test_demo_up_llm.py`:

```python
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/cloud/demo_up_llm.sh"
# create_secrets.sh + smoke_cloud.sh require NGINX_API_KEY; deploy.sh (llm backend)
# requires LLM_BASE_URL/LLM_MODEL, which demo_up_llm.sh sets itself.
ENV = {"PATH": "/usr/bin:/bin", "GCP_PROJECT": "demo", "NGINX_API_KEY": "test-key"}


def test_script_parses_with_bash_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_has_strict_mode_and_dry_run():
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "--dry-run" in text


def test_dry_run_chains_the_building_blocks_in_order():
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=ENV
    )
    assert out.returncode == 0, out.stderr
    o = out.stdout
    prov = o.index("==> provision Standard zonal cluster")
    kserve = o.index("==> install cert-manager + KServe")
    secrets = o.index("==> create gateway + LLM secrets")
    deploy = o.index("==> deploy charts (llm backend)")
    smoke = o.index("==> smoke test")
    assert prov < kserve < secrets < deploy < smoke
    # routes the api at the in-cluster predictor with the llm backend
    assert "env.modelBackend=llm" in o
    assert "medical-qa-kserve-predictor.medical-qa.svc.cluster.local/v1" in o
    # provisions a ZONAL cluster (not the regional default) — free-trial cost guard
    assert "--location asia-southeast1-a" in o
    # waits for the InferenceService before smoke
    assert "wait --for=condition=Ready isvc/medical-qa-kserve" in o
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/cloud/test_demo_up_llm.py -v`
Expected: FAIL (script file does not exist).

- [ ] **Step 3: Create the orchestrator**

Create `scripts/cloud/demo_up_llm.sh`:

```bash
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
```

- [ ] **Step 4: Make it executable and run the test**

Run:
```bash
chmod +x scripts/cloud/demo_up_llm.sh
.venv/bin/pytest tests/cloud/test_demo_up_llm.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/cloud/demo_up_llm.sh tests/cloud/test_demo_up_llm.py
git commit -m "feat(cloud): demo_up_llm.sh one-command local in-cluster LLM bring-up"
```

---

### Task 7: Add `llm_host: in-cluster` path to `demo-up.yml`

**Files:**
- Modify: `.github/workflows/demo-up.yml`
- Test: `tests/deploy/test_cloud_workflows.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/deploy/test_cloud_workflows.py`:

```python
def test_demo_up_has_llm_host_incluster_path():
    text = (WF / "demo-up.yml").read_text()
    wf = _load("demo-up.yml")
    inputs = _triggers(wf)["workflow_dispatch"]["inputs"]
    assert inputs["llm_host"]["type"] == "choice"
    assert inputs["llm_host"]["default"] == "dgx"
    assert set(inputs["llm_host"]["options"]) == {"dgx", "in-cluster"}
    # in-cluster topology branch
    assert "scripts/cloud/provision_gke_standard.sh" in text
    assert "scripts/cloud/install_kserve.sh" in text
    assert "medical-qa-kserve-predictor" in text
    assert "isvc/medical-qa-kserve" in text
    # the Autopilot path is preserved (guarded, not removed)
    assert "scripts/cloud/provision_gke.sh" in text
    # get-credentials is now topology-driven, not hardcoded to medical-qa
    assert "cluster_name: ${{ env.GKE_CLUSTER }}" in text
    assert "location: ${{ env.GKE_LOCATION }}" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/deploy/test_cloud_workflows.py::test_demo_up_has_llm_host_incluster_path -v`
Expected: FAIL (`llm_host` input absent).

- [ ] **Step 3: Add the `llm_host` input**

In `.github/workflows/demo-up.yml`, under `inputs:` (after the `backend:` input block), add:

```yaml
      llm_host:
        description: Where the LLM runs when backend=llm (dgx = DGX vLLM via Cloudflare Tunnel; in-cluster = KServe llama.cpp on GKE Standard)
        required: true
        type: choice
        options:
          - dgx
          - in-cluster
        default: dgx
```

- [ ] **Step 4: Add the topology + provisioning steps**

In `.github/workflows/demo-up.yml`, replace the block that currently reads:

```yaml
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
```

with:

```yaml
      - uses: google-github-actions/setup-gcloud@v2

      - name: Set cluster topology from llm_host
        run: |
          if [ "${{ inputs.llm_host }}" = "in-cluster" ]; then
            {
              echo "GKE_CLUSTER=medical-qa-llm"
              echo "GKE_LOCATION=asia-southeast1-a"
              echo "LLM_BASE_URL=http://medical-qa-kserve-predictor.${K8S_NAMESPACE}.svc.cluster.local/v1"
              echo "LLM_MODEL=qwen2.5-1.5b-instruct"
            } >> "$GITHUB_ENV"
          else
            {
              echo "GKE_CLUSTER=medical-qa"
              echo "GKE_LOCATION=asia-southeast1"
            } >> "$GITHUB_ENV"
          fi

      - name: Provision GKE Autopilot (regional)
        if: inputs.llm_host != 'in-cluster'
        run: bash scripts/cloud/provision_gke.sh

      - name: Provision GKE Standard (zonal)
        if: inputs.llm_host == 'in-cluster'
        run: bash scripts/cloud/provision_gke_standard.sh

      - uses: google-github-actions/get-gke-credentials@v2
        with:
          cluster_name: ${{ env.GKE_CLUSTER }}
          location: ${{ env.GKE_LOCATION }}

      - name: Install cert-manager + KServe (in-cluster LLM)
        if: inputs.llm_host == 'in-cluster'
        run: bash scripts/cloud/install_kserve.sh

      - name: Create nginx API-key secret
        run: bash scripts/cloud/create_secrets.sh

      - name: Deploy charts
        run: bash scripts/cloud/deploy.sh

      - name: Wait for InferenceService (in-cluster LLM)
        if: inputs.llm_host == 'in-cluster'
        run: .tools/bin/kubectl -n "$K8S_NAMESPACE" wait --for=condition=Ready isvc/medical-qa-kserve --timeout=600s

      - name: Wait for nginx then smoke-test
        run: |
          .tools/bin/kubectl -n "$K8S_NAMESPACE" rollout status deploy/medical-qa-nginx --timeout=300s || true
          bash scripts/cloud/smoke_cloud.sh
```

(The job-level `env:` block is unchanged — `LLM_BASE_URL`/`LLM_MODEL` keep their `${{ vars.* }}` defaults for the `dgx` path; the in-cluster step overrides them via `$GITHUB_ENV`.)

- [ ] **Step 5: Run the full workflow test file**

Run: `.venv/bin/pytest tests/deploy/test_cloud_workflows.py -v`
Expected: PASS — the new test plus all existing ones (`test_demo_up_is_dispatch_and_uses_oidc_and_scripts`, `test_demo_up_wires_llm_backend_toggle`, etc.), which still find every literal they assert.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/demo-up.yml tests/deploy/test_cloud_workflows.py
git commit -m "feat(ci): demo-up llm_host in-cluster path (Standard zonal + KServe)"
```

---

### Task 8: Add `llm_host` teardown branch to `demo-down.yml`

**Files:**
- Modify: `.github/workflows/demo-down.yml`
- Test: `tests/deploy/test_cloud_workflows.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/deploy/test_cloud_workflows.py`:

```python
def test_demo_down_has_llm_host_topology_teardown():
    text = (WF / "demo-down.yml").read_text()
    wf = _load("demo-down.yml")
    inputs = _triggers(wf)["workflow_dispatch"]["inputs"]
    assert inputs["llm_host"]["type"] == "choice"
    assert inputs["llm_host"]["default"] == "dgx"
    assert set(inputs["llm_host"]["options"]) == {"dgx", "in-cluster"}
    # tears down the cluster the topology selected
    assert 'get-credentials "$GKE_CLUSTER" --location "$GKE_LOCATION"' in text
    assert "scripts/cloud/teardown.sh" in text
    # bucket-delete escape hatch preserved
    assert "delete_bucket" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/deploy/test_cloud_workflows.py::test_demo_down_has_llm_host_topology_teardown -v`
Expected: FAIL (`llm_host` input absent in demo-down).

- [ ] **Step 3: Add the `llm_host` input**

In `.github/workflows/demo-down.yml`, under `inputs:`, add the `llm_host` block *before* `delete_bucket`:

```yaml
      llm_host:
        description: Which demo cluster to tear down (dgx = Autopilot medical-qa; in-cluster = Standard medical-qa-llm)
        required: true
        type: choice
        options:
          - dgx
          - in-cluster
        default: dgx
```

- [ ] **Step 4: Add topology step + make teardown location-aware**

In `.github/workflows/demo-down.yml`, replace:

```yaml
      - uses: google-github-actions/setup-gcloud@v2

      - name: Get cluster credentials
        continue-on-error: true
        run: gcloud container clusters get-credentials medical-qa --region asia-southeast1

      - name: Release LoadBalancer and delete cluster
        run: bash scripts/cloud/teardown.sh
```

with:

```yaml
      - uses: google-github-actions/setup-gcloud@v2

      - name: Set cluster topology from llm_host
        run: |
          if [ "${{ inputs.llm_host }}" = "in-cluster" ]; then
            { echo "GKE_CLUSTER=medical-qa-llm"; echo "GKE_LOCATION=asia-southeast1-a"; } >> "$GITHUB_ENV"
          else
            { echo "GKE_CLUSTER=medical-qa"; echo "GKE_LOCATION=asia-southeast1"; } >> "$GITHUB_ENV"
          fi

      - name: Get cluster credentials
        continue-on-error: true
        run: gcloud container clusters get-credentials "$GKE_CLUSTER" --location "$GKE_LOCATION"

      - name: Release LoadBalancer and delete cluster
        run: bash scripts/cloud/teardown.sh
```

(`teardown.sh` reads `GKE_CLUSTER`/`GKE_LOCATION` from the environment that the topology step wrote to `$GITHUB_ENV`.)

- [ ] **Step 5: Run the full workflow test file**

Run: `.venv/bin/pytest tests/deploy/test_cloud_workflows.py -v`
Expected: PASS — the new test plus the existing `test_demo_down_is_dispatch_tears_down_and_optionally_deletes_bucket`.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/demo-down.yml tests/deploy/test_cloud_workflows.py
git commit -m "feat(ci): demo-down llm_host branch tears down the right cluster"
```

---

## Final verification (after all tasks)

- [ ] **Run the full suite:**

Run: `.venv/bin/pytest -q`
Expected: all pass (the previous baseline was 235 passed / 1 skipped; this adds ~13 tests across `tests/cloud/` + `tests/deploy/`).

- [ ] **Sanity-check the local orchestrator dry-run end-to-end:**

Run: `GCP_PROJECT=demo NGINX_API_KEY=x bash scripts/cloud/demo_up_llm.sh --dry-run`
Expected: prints the ordered chain (provision → install_kserve → create_secrets → deploy → isvc wait → smoke) with no real cloud calls.

- [ ] **Confirm `gcloud --location` accepts a zone** (the linchpin of Task 2) — quick doc/CLI check before the first real run:

Run: `gcloud container clusters delete --help | grep -A2 -- --location`
Expected: help text states `--location` accepts a zone or region (compute zone/region). (If your gcloud is too old to support `--location`, fall back to `--zone`/`--region` per topology — note it on the PR.)

---

## Self-review notes (author)

- **Spec coverage:** §1 scripts → Tasks 3 (provision_gke_standard), 4 (install_kserve), 6 (demo_up_llm); §1 config/teardown → Tasks 1, 2; §2 nginx fix → Task 5; §3 workflows → Tasks 7 (demo-up), 8 (demo-down); §4 testing → each task's test + Final verification; §5 IAM → Operator prerequisites. `deploy.sh`/`create_secrets.sh`/`smoke_cloud.sh` deliberately untouched (spec "Unchanged").
- **No-placeholder check:** every code step shows full file content or an exact before/after block; every test step has runnable code + expected result.
- **Name consistency:** `GKE_CLUSTER`/`GKE_LOCATION`/`GKE_ZONE`/`CERT_MANAGER_VERSION`/`KSERVE_VERSION` are defined in Task 1 and used verbatim in Tasks 2–8; the in-cluster cluster is `medical-qa-llm` and zone `asia-southeast1-a` everywhere; predictor URL `medical-qa-kserve-predictor.<ns>.svc.cluster.local/v1` matches the validated Path A.
