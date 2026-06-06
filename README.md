# Medical_QA_MLOps

Runtime inference package for the Medical QA MLOps platform.

## Repository Boundary

This repository is self-contained for runtime API and retrieval-service code.
It does not import from a sibling `baseline/` checkout.

The production KG retrieval backend expects local artifact files, usually
populated later by DVC or mounted into the retrieval container:

- `index_hyperedge.bin`
- `index_entity.bin`
- `hedge_ids.npy`
- `entity_names.npy`
- `medical_hg.json`

Set `KG_DATA_DIR` to the directory containing those files. Heavy retrieval
dependencies are optional and installed with:

```bash
pip install ".[runtime]"
```


## Smoke MLOps Pipeline

Plan 2 adds a self-contained smoke pipeline. It does not depend on a sibling
`baseline/` checkout.

Install pipeline dependencies:

```bash
make install-pipeline
```

Run the pipeline locally without DVC orchestration:

```bash
make smoke-pipeline-local
```

Run the same stages through DVC:

```bash
make smoke-pipeline
```

Dry-run MLflow registration:

```bash
make mlflow-register-dry-run
```

The smoke pipeline writes artifacts under `artifacts/smoke/`. DVC tracks stage
lineage through `dvc.yaml`; MLflow receives metrics and artifacts through
`mlops/mlflow_register.py`.

## Docker, Helm, KServe, and CI

Plan 3 adds deployable app-serving artifacts for the runtime package:

- `docker/api.Dockerfile` builds the FastAPI pre/post-processing API.
- `docker/retrieval.Dockerfile` builds the KG retrieval service with the `runtime`
  extra, but does not bake KG artifacts or encoder weights into the image.
- `docker/kserve-mock.Dockerfile` builds the CPU KServe mock predictor.
- `docker/pipeline-init.Dockerfile` builds the DVC-capable init image used by the
  retrieval Helm chart to run `dvc pull`.

Install local Helm and kubectl tools:

```bash
make install-deploy-tools
```

Lint and render all charts:

```bash
make helm-lint
make helm-template
```

Run client-side dry-run validation for built-in Kubernetes resources:

```bash
make helm-dry-run
```

The KServe chart is structurally tested locally because `kubectl --dry-run=client`
cannot validate `serving.kserve.io` resources without the KServe CRD installed in a
cluster.

Build images on an x86 CI runner, or locally through `buildx` when needed:

```bash
make docker-build
```

The development host may be `aarch64`, while the target GKE nodes are `linux/amd64`.
The Docker build target and GitHub Actions workflow therefore pin
`--platform linux/amd64`.

The Helm charts cover API, retrieval, NGINX API-key gateway, and KServe mock
`InferenceService`. Live GKE deployment, GCS DVC remote credentials, a self-hosted
configuration, and observability stacks are deferred to Plan 4.

## Cloud Deploy — GKE-only Demo (slim Plan 4)

Deploy the stack to GKE Autopilot with one-click GitHub workflows and keyless
CI/CD. The CI demo uses the **mock** backend; flip to `MODEL_BACKEND=vllm`
(self-hosted vLLM on the DGX-Spark via Cloudflare Tunnel) for a real model —
see `docs/runbooks/dgx-vllm-cloudflare.md`. Retrieval is always real. See the
full runbook in [`docs/cloud-setup.md`](docs/cloud-setup.md):

- **Demo Up (GKE)** / **Demo Down (GKE)** — manual workflows to bring the demo up
  (provision + deploy + smoke) and tear it down (release LB + delete cluster).
- **Auto Deploy** — pushes to `main` auto-roll new images while the cluster is up,
  and skip green when it's down.

One-time bootstrap (`setup_gcs_dvc_remote.sh`, `setup_workload_identity.sh`,
`setup_github_oidc.sh`) sets up the GCS DVC remote and keyless GitHub→GCP auth.
Live GKE deploy in `asia-southeast1`.
