# Cloud Setup Runbook — GKE-only Demo + Automated CI/CD (slim Plan 4)

Deploy the Medical QA stack to a live GKE Autopilot cluster with a cheap,
ephemeral, automated workflow. The model is served by the **mock** backend —
this demonstrates the full request plumbing, real KG retrieval
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

   > **Important:** `GITHUB_REPO` must match your repository's exact `owner/repo` slug **including casing** — it is embedded in the OIDC provider's attribute-condition and the principalSet binding. A casing mismatch makes every workflow's GCP auth fail with an opaque permission error.

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
- **Real 3B model (vLLM):** the deploy workflows default to the `vllm` backend.
  Set in GitHub repo settings: **Secret** `LLM_API_KEY` (the DGX vLLM api-key);
  **Vars** `LLM_BASE_URL` (e.g. `https://llm.<domain>/v1` — must end in `/v1`) and
  `LLM_MODEL` (e.g. `medical-qa-llama-gdpo`); optional Var `MODEL_BACKEND` to
  override the default. `Demo Up (GKE)` has a `backend` input (`vllm`|`mock`,
  default `vllm`); `Auto Deploy` uses `vllm` unless `MODEL_BACKEND` is set. Stand
  up the DGX server first — see [`runbooks/dgx-vllm-cloudflare.md`](runbooks/dgx-vllm-cloudflare.md).
  Run `Demo Up` with `backend: mock` to demo the plumbing when the DGX is offline.
- **Note:** Keep the `Demo Up` `namespace` input at the default `medical-qa`. The retrieval pod's Workload Identity binding (`setup_workload_identity.sh`) is namespace-scoped; deploying into a different namespace requires re-running that bootstrap script for the new namespace, or the keyless `dvc pull` will fail.
