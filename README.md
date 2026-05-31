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
