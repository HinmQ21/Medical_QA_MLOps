# Contract-Version Surfacing (SP4) — Design

**Remainder of the serving-alignment sub-project.** The encoder-config wiring was
folded into SP2; this adds the small remaining piece: the serving stack reports
which retrieval contract it is running, for parity traceability against the
model version registered by the SP3 pipeline.

## Goal

Expose `RETRIEVAL_CONTRACT_VERSION` (and the actually-configured encoder/KG) from
the running serving stack, so an operator (or the MLflow registry correspondence
check) can confirm a deployed service is serving the contract a model was
registered under. Report only — no enforcement.

## Context

- `RETRIEVAL_CONTRACT_VERSION` lives in `src/medical_qa_platform/retrieval/contract.py`
  (currently `v1-medembed-small` after SP2).
- The API (`api/app.py`) exposes `/health`, `/ready`, `/predict`, `/metrics`;
  `PredictResponse` carries `answer, evidence, backend, model_version, latency_ms,
  trace_id`. The retrieval service (`retrieval/service.py`) exposes `/health`,
  `/ready`, `/search`, `/metrics`.
- SP3 registers each model version in MLflow with a `retrieval_contract_version`
  tag; this closes the loop on the serving side.

## Changes

1. **`api/schemas.py`** — add `contract_version: str` to `PredictResponse`.
2. **`api/app.py`** —
   - `/predict`: set `contract_version=RETRIEVAL_CONTRACT_VERSION` (import from
     `..retrieval.contract`).
   - add `GET /version` → `{"contract_version": RETRIEVAL_CONTRACT_VERSION,
     "model_version": app.state.model_version, "backend": app.state.backend.name}`.
3. **`retrieval/service.py`** — add `GET /version` →
   `{"contract_version": RETRIEVAL_CONTRACT_VERSION, "encoder_model":
   os.environ.get("KG_ENCODER_MODEL", "abhinand/MedEmbed-small-v0.1"),
   "kg_data_dir": os.environ.get("KG_DATA_DIR", "data/")}`. Reads env so it
   reports the configured contract regardless of whether the backend is the real
   `KGRetrieval` or a test fake.

## Data Flow

`/version` is static-ish: it reads the `RETRIEVAL_CONTRACT_VERSION` constant plus
`app.state` / env. `/predict` gains one response field. No change to retrieval or
generation logic, and no change to `/search`.

## Testing

- API `/version` returns `contract_version == RETRIEVAL_CONTRACT_VERSION`,
  plus `model_version` and `backend`.
- `/predict` response includes `contract_version`.
- Retrieval `/version` returns `contract_version`, `encoder_model`, `kg_data_dir`
  (assert the encoder default is the small model; assert an overridden
  `KG_ENCODER_MODEL`/`KG_DATA_DIR` env is reflected).
- All via FastAPI `TestClient` with fakes — no GPU/heavy deps. `make test` green,
  coverage ≥ 80%.

## Out of Scope

- No contract **enforcement** (mismatch detection / refusing to serve) — report
  only.
- No change to `/search`, retrieval logic, or the model backends.
- The default encoder string is duplicated between `KGRetrieval` and the
  retrieval `/version` (both default to the small model); acceptable for a
  report endpoint. If this drifts later, centralize the default.

## Risks / Open Questions

- **`backend.name` availability:** `app.state.backend` is set at startup; if
  `/version` is hit before startup completes it could be unset. The endpoint
  reads `getattr(app.state, "backend", None)` defensively and reports
  `backend=None` in that window (same robustness as `/ready`).
