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
