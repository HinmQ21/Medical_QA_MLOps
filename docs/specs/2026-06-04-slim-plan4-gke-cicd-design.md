# MLOps Slim Plan 4 — GKE-only Demo + Automated CI/CD Design

**Slim variant of Plan 4 (`docs/specs/2026-06-01-p0-cloud-deploy-gke-runpod-design.md`).**
Same goal — deploy the `medical_qa_platform` app-serving stack to a live GKE
Autopilot cluster — but optimized for a **cheap, ephemeral demo**: no RunPod
(model served by the CPU **mock** backend), and the deploy path is **automated
through GitHub Actions** with keyless auth, rather than a manual `workflow_dispatch`
with placeholder credentials.

## Goal

Make the stack demoable on real GKE with a one-click bring-up and tear-down, plus
true continuous deployment while the cluster is alive — at minimum cost. Concretely:

- A **`demo-up`** GitHub workflow provisions GKE, deploys the four Helm charts
  (api on the **mock** backend, no RunPod), runs a smoke test, and prints the
  public LoadBalancer IP.
- While the cluster is alive, **every push to `main`** auto-builds images (existing
  CI) and **auto-deploys** them (rolling `helm upgrade`); a push made while the
  cluster is down **skips the deploy and stays green** (no red runs).
- A **`demo-down`** workflow tears the cluster down (stopping Autopilot +
  LoadBalancer billing) while keeping the cheap durable resources (GCS bucket,
  GSAs, Workload Identity Federation) so the next demo is fast and needs no
  re-bootstrap.
- GKE → GCP auth is **keyless** via GitHub OIDC → GCP Workload Identity Federation
  (no long-lived SA keys in repo secrets).

## Context

- **Plans 1–3 are done and merged to `main`.** Plan 1 = runtime package; Plan 2 =
  DVC + MLflow smoke pipeline; Plan 3 = Docker/Helm/KServe/CI. SP1–SP4 brought the
  retrieval ranker to train↔serve parity, swapped the encoder to MedEmbed-small,
  added the full MLflow pipeline, and surfaced the contract version.
- **CI** (`.github/workflows/ci.yml`) already builds four `linux/amd64` images on
  every push to `main` and pushes them to `ghcr.io/hinmq21/medical-qa-{api,retrieval,kserve-mock,pipeline-init}`
  (`:sha` + `:latest`, publicly pullable — no imagePullSecret needed).
- The api backend is **pluggable** (`MODEL_BACKEND`): committed defaults are
  `mock` (`deploy/helm/api/values.yaml`), with `kserve` and `runpod` as opt-ins.
  GKE-only demo uses **mock**; RunPod stays a future flip (one config + one secret).
- The **full Plan 4** (`plan4-cloud-deploy` branch) specced six `scripts/cloud/`
  scripts + chart prod overlays + a runbook, but is **docs-only (0 tasks
  implemented)** and leaves `deploy.yml` auth as a placeholder. This slim plan
  reuses most of that script/chart work, drops the RunPod path, and replaces the
  manual deploy with automated workflows.

## What this demo proves (and does not)

GKE-only with the mock backend is an **infrastructure / plumbing demo**, not a
model-quality demo:

| Demonstrated | Not demonstrated |
|---|---|
| Full request path: nginx (x-api-key) → api → **real KG retrieval** (MedEmbed-small, CPU) → contract surfacing → `/predict` + `/version` | Real 3B model reasoning — answers come from the **mock** backend |
| Keyless GCS artifact pull (Workload Identity), Autopilot autoscaling, health/ready | GPU inference |
| Keyless GitHub→GCP CD, ephemeral cost control | — |

The **retrieval path is real** (the SP1–SP4 parity story): the retrieval pod pulls
the real MedEmbed-small KG (~56.8K hyperedges) from GCS and serves the 8-term
fusion ranker. Only the *answer letter* is mocked.

## Decisions (locked during brainstorming)

1. **CI/CD trigger = demo-up/down + auto-deploy-while-alive** (not auto-on-every-push,
   not pure-manual). Rationale: push-triggered CD fires on the *push* event, not on
   *cluster-up*, so "push then turn on GCP" would deploy nothing and leave red runs.
   Auto-deploy is meaningful only while the cluster lives; bring-up is its own
   trigger.
2. **No RunPod.** api serves the mock backend (base chart values). No RunPod
   secret, no api prod overlay.
3. **KG delivery = GCS + Workload Identity + `dvc pull`** (faithful to Plan 4 and to
   the retrieval-parity story). The retrieval initContainer pulls the real
   MedEmbed-small KG keylessly.
4. **Keyless Actions→GCP** via Workload Identity Federation (GitHub OIDC). No SA
   keys committed; the WIF provider name + deploy GSA email live in GitHub repo
   **vars**.
5. **One-time bootstrap vs recurring demo split** (least privilege + fast demos):
   - **Bootstrap (operator, run once from Cloud Shell / a machine with owner):**
     enable APIs, create the GCS bucket + first `dvc push` of the KG, create the
     retrieval GSA + objectViewer + Workload-Identity binding, and set up the
     GitHub WIF pool/provider + a **deploy GSA**. These create **durable** resources
     that persist across demos.
   - **Recurring (GitHub Actions, deploy GSA, least privilege):** `demo-up`
     provisions the cluster + nginx secret + deploys + smokes; `deploy.yml`
     rolls updates while alive; `demo-down` deletes the cluster. None of these
     touch IAM/bucket/GSA — they assume bootstrap ran.
6. **Teardown keeps cheap durable state.** `demo-down` deletes the cluster (and
   thus the LoadBalancer) but **keeps** the bucket, GSAs, and WIF. The KG is not
   re-pushed on the next demo. An optional `delete_bucket` input (default `false`)
   allows a full clean-up.
7. **Region/namespace defaults** `asia-southeast1` / `medical-qa`; all overridable
   via env/`config.sh`/repo vars.

## Architecture & Data Flow

```
Internet ──► nginx LoadBalancer (public IP, x-api-key auth)
                 └─► api Service (ClusterIP:8000, MODEL_BACKEND=mock)
                       ├─► retrieval Service (ClusterIP:8001) ──► PVC (KG artifacts)
                       │        ▲ initContainer: dvc pull (keyless, Workload Identity)
                       │        └──────── GCS bucket gs://<project>-medical-qa-dvc (durable)
                       └─► mock backend (in-process; deterministic answer letter)
   KServe CPU-mock InferenceService = separate smoke/demo (off the request path)

CI/CD:
  push main ─► CI (test + build + push GHCR :sha/:latest)
                     └─workflow_run(completed, main)─► deploy.yml
                                                          │ cluster reachable?
                                                  no ──► log + exit 0 (green)
                                                  yes ─► helm upgrade (rolling) + smoke
  [demo-up]   workflow_dispatch → auth(OIDC) → provision_gke → create_secrets(nginx)
                                  → deploy.sh(mock) → smoke_cloud → print LB IP
  [demo-down] workflow_dispatch → auth(OIDC) → helm uninstall nginx → delete cluster
                                  (keep bucket/GSA/WIF unless delete_bucket=true)
```

## Components & Deliverables

### Scripts — `scripts/cloud/` (idempotent, `set -euo pipefail`, `--dry-run`)

| Script | Phase | Responsibility |
|--------|-------|----------------|
| `config.sh` | shared | Sourced defaults: region, cluster, namespace, bucket, REGISTRY, IMAGE_TAG, and WIF vars (`WIF_POOL`, `WIF_PROVIDER`, `DEPLOY_GSA`). **No RunPod vars.** |
| `provision_gke.sh` | recurring | Create the Autopilot cluster; `get-credentials`. *(Reused from Plan 4.)* |
| `setup_gcs_dvc_remote.sh` | bootstrap | Create the GCS bucket; `dvc remote add`; `dvc push` the KG. *(Reused.)* |
| `setup_workload_identity.sh` | bootstrap | Retrieval GSA + `storage.objectViewer` + `iam.workloadIdentityUser` binding to `[medical-qa/medical-qa-retrieval]`. *(Reused.)* |
| `setup_github_oidc.sh` | bootstrap | **NEW.** Create the WIF pool + an OIDC provider scoped to the GitHub repo; create the **deploy GSA**; grant it least-privilege roles (`container.admin` for cluster create/delete + helm/kubectl; `serviceusage.serviceUsageConsumer`); bind the repo principal → deploy GSA via `iam.workloadIdentityUser`. Prints the provider resource name + GSA email for GitHub repo vars. |
| `create_secrets.sh` | recurring | **nginx API key only** (`medical-qa-nginx-api-key`). RunPod secret removed. |
| `deploy.sh` | recurring | `helm upgrade --install` retrieval (WI SA + bucket), api (**base/mock**, no overlay), nginx (`values-prod.yaml`: LoadBalancer + existingSecret), kserve. |
| `smoke_cloud.sh` | recurring | Resolve the nginx LB IP; `/health`; authed `POST /predict` (assert a valid answer letter from the mock); `GET /version` (assert `contract_version`). *(Reused.)* |

Each `--dry-run` prints the exact command sequence and makes no cloud calls.

### Helm chart overlays (committed base values stay mock/dev-safe)

- **retrieval:** add a templated `ServiceAccount` with the optional
  `iam.gke.io/gcp-service-account` Workload-Identity annotation; replace the
  placeholder `dvc.config` with a `dvc.bucket` value used to build the dvc-config
  ConfigMap; set `serviceAccountName` on the pod. *(Plan 4 Task 7.)*
- **nginx:** add `auth.existingSecret` and `service.type`; `secret.yaml` is skipped
  when `existingSecret` is set; `values-prod.yaml` = `LoadBalancer` + `existingSecret`.
  *(Plan 4 Task 8.)*
- **api:** **no prod overlay.** Base `values.yaml` already sets `MODEL_BACKEND=mock`
  with no secret refs. (The RunPod overlay from full Plan 4 is intentionally omitted.)

### Workflows — `.github/workflows/`

All three authenticate with `google-github-actions/auth@v2` (WIF, keyless) +
`google-github-actions/get-gke-credentials@v2`, reading the WIF provider + deploy
GSA from repo **vars**.

- **`demo-up.yml`** (`workflow_dispatch`, inputs: `image_tag` default `latest`,
  `namespace` default `medical-qa`): auth → `provision_gke.sh` → `create_secrets.sh`
  (nginx) → `deploy.sh` → `smoke_cloud.sh` → echo the LoadBalancer IP. Assumes the
  bootstrap resources (bucket, GSAs, WIF) exist.
- **`deploy.yml`** (`on: workflow_run` of the CI workflow, `completed`, branch
  `main`): auth → `gcloud container clusters describe` to test reachability →
  **if absent: log "cluster down — skipping deploy" and exit 0**; **if present:**
  `helm upgrade` the four charts (rolling) + `smoke_cloud.sh`. This is the
  auto-deploy-while-alive path; pushes made while the cluster is down do not fail.
- **`demo-down.yml`** (`workflow_dispatch`, input `delete_bucket` default `false`):
  auth → `helm uninstall medical-qa-nginx` (release the LoadBalancer) → delete the
  Autopilot cluster → optionally delete the bucket. Keeps GSAs + WIF.

### Docs

- **`docs/cloud-setup.md`** — runbook: one-time bootstrap (incl. `setup_github_oidc.sh`
  and the GitHub repo vars to set), the demo-up/demo-down workflow buttons, the
  auto-deploy-while-alive behavior, cost notes, and teardown.
- **README** — a pointer section to the runbook.

## Identity & Least Privilege

- **Retrieval GSA** (`medical-qa-retrieval@…`): `roles/storage.objectViewer` on the
  bucket only; used keylessly by the dvc-pull initContainer via WI.
- **Deploy GSA** (used by all three workflows): `roles/container.admin` (create/delete
  clusters; full helm/kubectl in-cluster) + `roles/serviceusage.serviceUsageConsumer`.
  It does **not** get storage-admin or IAM-admin — bucket/GSA/WIF are created once by
  the operator during bootstrap, so the CI identity cannot mutate IAM or data.
- **No SA keys** anywhere; the deploy GSA is reachable only from the configured
  GitHub repo via the OIDC provider's attribute condition (`repository ==` the repo).

## Testing Strategy / Completion Gates (no cloud calls)

- **Script structural tests** (pytest reading file text + `bash -n` + `--dry-run`):
  every script asserts `set -euo pipefail`, the required gcloud/kubectl/helm/dvc
  commands and key flags, and a working `--dry-run`. `create_secrets.sh` creates
  **only** the nginx secret (no RunPod). `setup_github_oidc.sh` creates the pool,
  the provider with the repo attribute condition, the deploy GSA, the role grants,
  and the workloadIdentityUser binding.
- **Chart render tests** (helm template + parse): retrieval renders the SA + WI
  annotation + real bucket; nginx prod renders `LoadBalancer` + `existingSecret`
  with no literal `change-me-dev-key`; **api base still renders `MODEL_BACKEND=mock`
  and no secret ref** (asserting RunPod was not introduced).
- **Workflow YAML tests** (parse with PyYAML): all three use `auth@v2` +
  `get-gke-credentials`; `deploy.yml` triggers on `workflow_run` (CI, main) and
  contains the cluster-reachability skip; `demo-up`/`demo-down` invoke the right
  scripts; `demo-down` uninstalls nginx before deleting the cluster.
- **Runbook test**: `docs/cloud-setup.md` documents bootstrap, `setup_github_oidc.sh`,
  the GitHub vars, demo-up, demo-down, teardown, and cost; README points to it.
- **Gates:** `make test` green with coverage ≥ 80%; `make helm-lint` and
  `make helm-template` clean with the prod overlays; `shellcheck` if available.

## Out of Scope

- **RunPod / real 3B serving** — a future flip (`modelBackend: runpod` + RunPod
  secret + api overlay); the full Plan 4 already specs it.
- **Plan 5 items** — Prometheus/Grafana/Loki/Tempo, MLflow tracking-server deploy,
  Evidently drift dashboard, HPA load demo, HTTPS/custom domain + GKE Ingress.
- **Always-on cluster / GitOps controller (ArgoCD/Flux)** — rejected for an
  ephemeral cost-optimized demo.

## Risks / Open Questions

- **Cost:** Autopilot pods + one public LoadBalancer accrue charges while the
  cluster is up. The runbook mandates `demo-down`; an orphaned LoadBalancer keeps
  billing, so `demo-down` uninstalls nginx before deleting the cluster.
- **demo-up latency:** first request waits on cluster create + `dvc pull` of the KG
  (~480 MB) into the PVC + encoder load + cold start. The smoke step must tolerate
  warm-up; demos are not instant.
- **Mock answers:** the demo must be presented as infrastructure, not model quality.
- **OIDC attribute condition:** the provider must restrict to this exact repo (and
  optionally `ref`), or any repo in the org could assume the deploy GSA.
- **`workflow_run` scoping:** `deploy.yml` must filter to `head_branch == main` and
  `conclusion == success` so it only deploys images that CI actually built+pushed.
- **Bootstrap drift:** if the operator deletes the bucket/GSA, `demo-up` fails until
  bootstrap is re-run; the runbook calls this out.
