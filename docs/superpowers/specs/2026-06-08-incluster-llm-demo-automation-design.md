# Design: Automate the in-cluster LLM demo bring-up ("Path B")

**Date:** 2026-06-08
**Status:** Approved (brainstorming) — pending spec review before planning

## Problem

Standing up the **in-cluster LLM** serving path (GKE Standard zonal + KServe
RawDeployment + llama.cpp Qwen2.5-1.5B InferenceService, reached via the `llm`
backend) was validated **manually** ("Path A") from Google Cloud Shell. Three pieces
of that bring-up are **not scripted**, so re-running it means copy-pasting ~8 commands
by hand every time:

1. `scripts/cloud/provision_gke.sh` only creates a GKE **Autopilot regional** cluster
   (`gcloud container clusters create-auto`), which ships **without** the KServe CRD.
   There is no script to create the **Standard zonal** cluster the in-cluster path needs.
2. There is **no script to install cert-manager + KServe** (CRD chart `kserve-crd` +
   controller chart `kserve-resources`, RawDeployment mode). These were `helm install`-ed
   by hand, hitting two gotchas: KServe v0.17+ renamed the controller chart
   `kserve`→`kserve-resources`, and Helm 4 will not auto-resolve "latest" for OCI charts
   (must pass `--version`).
3. The nginx gateway chart has a **latent map_hash bug**: an API key ≳40 chars overflows
   the default `map_hash_bucket_size 64` and nginx crashloops with
   `[emerg] could not build map_hash`. Path A dodged it by using a short 32-char key.

The existing one-click GitHub workflows (`demo-up.yml` / `demo-down.yml`) only know the
Autopilot-regional topology. `demo-up.yml` already has a `backend: llm | mock` input, but
`backend=llm` there means "point at the **DGX vLLM** server over a Cloudflare Tunnel"
(`LLM_BASE_URL` from a repo Var) — a different topology from the in-cluster predictor.

## Goal

Make the in-cluster LLM demo reproducible **without manual steps**, in two layers:

- **Engine (scripts):** idempotent scripts that provision the Standard zonal cluster,
  install cert-manager + KServe, and chain the whole bring-up — runnable locally
  (Cloud Shell) for one-command use and for debugging the CI path.
- **One-click (workflow):** extend `demo-up.yml` / `demo-down.yml` with an `llm_host`
  input so a maintainer can bring the in-cluster LLM demo up and down from the GitHub UI,
  reusing the existing keyless OIDC/WIF auth.

Plus the real fix for the nginx map_hash bug so API-key length no longer matters.

## Decisions (from brainstorming)

- **Scope = "B on A":** the heavy logic lives in **scripts**; the workflow only
  orchestrates and branches. (Chosen over inlining helm/KServe steps into YAML, and over
  full Terraform/GitOps which is overkill for a free-trial demo cluster.)
- **B lives in the existing workflows** (`demo-up.yml` + `demo-down.yml`) via a new
  `llm_host: dgx | in-cluster` input — **one entry point**. (Chosen over a separate
  `demo-up-llm.yml`.) The existing `dgx`/`mock` branches are wrapped under `if` and keep
  their exact current behavior; `llm_host` defaults to `dgx`.
- **Input name = `llm_host: dgx | in-cluster`** (`backend` says *what model*, `llm_host`
  says *where the LLM runs*; only meaningful when `backend=llm`). Chosen over
  `cluster_type: autopilot | standard-kserve`.
- **Include a local orchestrator `demo_up_llm.sh`** (Approach A) for one-command local
  bring-up and CI-failure debugging.
- **Fix the nginx bug for real** (`map_hash_bucket_size 128`) rather than keep the
  short-key workaround.

## Design

### 1. Script layer (engine) — `scripts/cloud/`

**New: `provision_gke_standard.sh`** — mirrors `provision_gke.sh`'s structure exactly
(same `--dry-run` handling, same idempotent describe-then-create guard, same
`gcloud services enable` block, ends with `get-credentials`), but creates a **Standard
zonal** cluster:

```bash
gcloud container clusters create "$GKE_CLUSTER" \
  --location "$GKE_LOCATION" \
  --num-nodes 1 --machine-type e2-standard-8 \
  --workload-pool="$GCP_PROJECT.svc.id.goog"
```

Idempotent: `gcloud container clusters describe "$GKE_CLUSTER" --location "$GKE_LOCATION"`
→ skip create if present. The caller sets `GKE_CLUSTER=medical-qa-llm` and
`GKE_LOCATION=$GKE_ZONE` so it never collides with the Autopilot `medical-qa`.

**New: `install_kserve.sh`** — installs the cluster-scoped prerequisites idempotently
(`helm upgrade --install`), with `--dry-run` support and pinned versions as config vars
(`CERT_MANAGER_VERSION` default `v1.20.2` — the known-good version from Path A;
`KSERVE_VERSION` default `v0.18.0`):

1. cert-manager: `helm repo add jetstack https://charts.jetstack.io` + `helm repo update`
   + `helm upgrade --install cert-manager jetstack/cert-manager -n cert-manager
   --create-namespace --version "$CERT_MANAGER_VERSION" --set crds.enabled=true`, then
   `kubectl -n cert-manager rollout status deploy/cert-manager-webhook`.
2. KServe CRD: `helm upgrade --install kserve-crd
   oci://ghcr.io/kserve/charts/kserve-crd --version "$KSERVE_VERSION"`.
3. KServe controller: `helm upgrade --install kserve
   oci://ghcr.io/kserve/charts/kserve-resources --version "$KSERVE_VERSION"
   -n kserve --create-namespace --set kserve.controller.deploymentMode=RawDeployment`,
   then `kubectl -n kserve rollout status deploy/kserve-controller-manager`.

Uses the repo's `$REPO_ROOT/.tools/bin/{helm,kubectl}` like the other cloud scripts.

**New (optional but included): `demo_up_llm.sh`** — thin local orchestrator that exports
the in-cluster topology (`GKE_CLUSTER=medical-qa-llm`, `GKE_LOCATION=$GKE_ZONE`,
`MODEL_BACKEND=llm`, `LLM_BASE_URL=<in-cluster predictor>`,
`LLM_MODEL=qwen2.5-1.5b-instruct`) and chains the existing building blocks:
`provision_gke_standard.sh` → `install_kserve.sh` → `create_secrets.sh` →
`deploy.sh` → wait for `isvc medical-qa-kserve` READY → `smoke_cloud.sh`. Supports
`--dry-run` (passes through to each sub-script). This is the one-command local path and
the way to reproduce/debug a CI failure in Cloud Shell.

**Changed: `config.sh`** — add two location vars (back-compat: defaults reproduce
today's regional Autopilot behavior):

```bash
: "${GKE_ZONE:=${GCP_REGION}-a}"      # default zone for the Standard cluster
: "${GKE_LOCATION:=$GCP_REGION}"      # region (Autopilot) or zone (Standard)
```

**Changed: `teardown.sh`** — replace `--region "$GCP_REGION"` with
`--location "$GKE_LOCATION"` on the `gcloud container clusters delete` call. `gcloud`
accepts a region **or** a zone for `--location`; the default `GKE_LOCATION=$GCP_REGION`
keeps the Autopilot teardown byte-for-byte equivalent. (Verified via `--dry-run` in the
plan.)

**Unchanged (already correct):**
- `deploy.sh` — already routes the api at `MODEL_BACKEND=llm` + `LLM_BASE_URL`/`LLM_MODEL`
  and installs the `kserve` chart **only when the CRD is present** (which it now is after
  `install_kserve.sh`), so the real Qwen2.5-1.5B InferenceService is created with no
  workflow change.
- `create_secrets.sh` — already idempotent; with the nginx fix below, key length no
  longer matters.
- `smoke_cloud.sh` — already backend-agnostic, polls the LB IP, and waits on each
  deployment's rollout (the api's `/ready` is gated on `backend.health_check()`, so its
  rollout implicitly waits for the predictor).
- `setup_workload_identity.sh` — unchanged.

### 2. nginx chart fix — `deploy/helm/nginx/templates/configmap.yaml`

Add `map_hash_bucket_size 128;` immediately before the `map $http_x_api_key` block:

```nginx
map_hash_bucket_size 128;
map $http_x_api_key $api_key_valid {
  default 0;
  "${API_KEY}" 1;
}
```

After this, any API-key length is safe; the short-key workaround is no longer required.

### 3. Workflow layer (B) — `.github/workflows/`

**`demo-up.yml`** — add an `llm_host` choice input (`dgx` | `in-cluster`, default `dgx`,
only meaningful when `backend=llm`). Add an early **"set topology"** step that derives,
based on `llm_host`, the env that the rest of the job reads:

| Var | `dgx` / `mock` (default, unchanged) | `in-cluster` |
|-----|--------------------------------------|--------------|
| `GKE_CLUSTER` | `medical-qa` | `medical-qa-llm` |
| `GKE_LOCATION` | `asia-southeast1` (region) | `asia-southeast1-a` (zone) |
| `LLM_BASE_URL` | `vars.LLM_BASE_URL` (DGX tunnel) | `http://medical-qa-kserve-predictor.<ns>.svc.cluster.local/v1` |
| `LLM_MODEL` | `vars.LLM_MODEL` | `qwen2.5-1.5b-instruct` |

Branching steps (the existing steps are guarded so the `dgx`/`mock` path is byte-for-byte
the same):

- **Provision:** `if in-cluster` → `provision_gke_standard.sh`; else → `provision_gke.sh`.
- **get-gke-credentials:** the existing `google-github-actions/get-gke-credentials@v2`
  step now reads `cluster_name: ${{ env.GKE_CLUSTER }}` and `location: ${{ env.GKE_LOCATION }}`
  (a value swap; defaults reproduce `medical-qa` / `asia-southeast1`).
- **Install KServe:** new step, `if in-cluster` → `bash scripts/cloud/install_kserve.sh`.
- **Create secrets / Deploy:** unchanged steps (`deploy.sh` installs the kserve chart
  because the CRD is now present and routes the api at the in-cluster predictor URL).
- **Wait for InferenceService:** new step, `if in-cluster` →
  `kubectl -n "$K8S_NAMESPACE" wait --for=condition=Ready isvc/medical-qa-kserve --timeout=600s`
  (GGUF download budget), before the smoke step.
- **Smoke / print endpoint:** unchanged.

**`demo-down.yml`** — add the same `llm_host` input; the "set topology" step derives
`GKE_CLUSTER` / `GKE_LOCATION` so `get-credentials` targets the right cluster and
`teardown.sh` (now `--location`-based) deletes the right one (zonal `medical-qa-llm` for
`in-cluster`, regional `medical-qa` otherwise).

### 4. Testing

- `tests/deploy/test_cloud_workflows.py` — extend to assert the new `llm_host` inputs and
  the `in-cluster` branch steps (set-topology, conditional provision, install-kserve,
  isvc-wait) exist in both workflows, and that the default keeps the Autopilot path.
- New assertion that the rendered nginx ConfigMap contains `map_hash_bucket_size 128;`
  (extend the existing nginx chart test, or add one under `tests/deploy/`).
- New scripts are exercised via their `--dry-run` mode (the established pattern for the
  other `scripts/cloud/*.sh`), asserting the right `gcloud`/`helm` command lines are
  printed (Standard `create --location`, the three KServe installs with `--version`, the
  orchestrator's ordered chain).

### 5. IAM (verify, not build)

`GCP_DEPLOY_SA` (the WIF deployer) already provisions Autopilot, so it should already hold
`roles/container.admin`; the identity that creates a GKE cluster automatically receives
cluster-admin Kubernetes RBAC, which is what cert-manager/KServe (cluster-scoped CRDs +
webhooks) require. The plan includes a one-line verification step
(`gcloud projects get-iam-policy` for the SA's `container.admin` binding) rather than a
new grant.

## Non-goals

- Full IaC (Terraform / Config Connector / ArgoCD / Flux) — overkill for a free-trial
  demo cluster spun up and down on demand.
- Auto Deploy (`deploy.yml`) targeting the in-cluster cluster — it stays on the Autopilot
  `medical-qa` regional cluster; this design only covers the manual `demo-up`/`demo-down`
  one-click path.
- A PVC for the GGUF cache (avoid re-download on pod restart) and pinning the
  `llama.cpp:server` image to a digest — documented follow-ups in
  `project_mlops_kserve_llamacpp`, out of scope here.
- Changing the `llm` backend code or the KServe chart contents (the chart already serves
  the real model; this work only automates installing KServe and bringing the demo up).

## Risks / notes

- **`gcloud ... --location` accepting both region and zone** is the linchpin of the
  `teardown.sh` change; confirmed by design intent and re-verified in the plan via
  `--dry-run` before relying on it.
- **KServe / cert-manager version pins** (`KSERVE_VERSION=v0.18.0`,
  cert-manager pinned) must be installed from the renamed chart coordinates
  (`kserve-resources`, not `kserve`) with explicit `--version` (Helm 4 OCI requirement).
- **InferenceService cold start:** the GGUF (`Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M`)
  downloads at pod start; the `isvc` wait uses a 600s timeout to cover it.
- **Blast radius on the proven path:** the `dgx`/`mock` branches are wrapped under `if`
  with unchanged scripts; the only shared edits are additive (`config.sh` new vars with
  back-compat defaults, `teardown.sh` region→location with a region default). The workflow
  structure test guards the default path.
- **Free-trial cost:** the Standard zonal cluster bills while up; `demo-down` with
  `llm_host: in-cluster` is the teardown. (`gcloud container clusters delete medical-qa-llm
  --location asia-southeast1-a`.)
