# Encoder Swap → MedEmbed-small (SP2) — Design

**Sub-project 2 of the train↔serve parity effort.** Switch the retrieval
contract's embedding encoder from `abhinand/MedEmbed-large-v0.1` (1024-d) to
`abhinand/MedEmbed-small-v0.1` (~384-d, BGE-small based) across the **whole
pipeline** (KG build → serving → golden), so one encoder is used everywhere and
there is no train↔serve skew by construction. Establish and verify the new
contract now; defer the (GPU-heavy, multi-day) 3-stage re-train to the operator.

## Goal

Make `MedEmbed-small-v0.1` the retrieval encoder of record: rebuild the KG index
with it, bump the retrieval contract version, regenerate the SP1 golden parity
fixture against it, wire it through serving config, and verify L1/L2 parity on
the new encoder — without running the multi-day re-train.

## Context

- **Why:** smaller encoder → lower CPU latency for the GKE retrieval service
  (serving runs the encoder on CPU). The user picked `abhinand/MedEmbed-small-v0.1`.
- **One encoder, no skew (locked in SP2 brainstorm):** the same encoder builds
  the KG index, serves queries, and (when retrained) trains the model. Mixing
  encoders would change the embedding space and break retrieval results.
- **Dimension change:** MedEmbed-small is ~384-d vs large's 1024-d, so the FAISS
  indices must be **rebuilt** (different dim) and the SP1 golden must be
  **regenerated** (different embeddings → different ranked results). The pure
  ranker logic (SP1 `ranker.py`) is encoder-agnostic and does **not** change.
- **Depends on:** SP1 (the ported `retrieve_v1` ranker + golden parity harness)
  and SP3 (the pipeline that the operator will use to re-train on the new KG).

## Decisions (locked during brainstorming)

1. **Scope now = build + verify the new contract; defer re-train.** The KG
   rebuild (CPU, ~30 min — does not touch the contended GPU) runs now; the
   3-stage re-train is operator-run later via the SP3 pipeline.
2. **Minimal back-compatible baseline edit (approved).** `baseline/scripts/serve/
   retrieval_tool.py` reads the encoder from `KG_ENCODER_MODEL` (default stays
   `abhinand/MedEmbed-large-v0.1`), so golden generation and future training can
   select the small encoder without changing default behavior. (Precedent: the
   SP1 runtime-package plan already made a single back-compat edit to this file.)
3. **Small KG lives under `mlops-platform/artifacts/kg_small/`** (approved) —
   generated, git-ignored. The large KG (`baseline/data`) is kept for rollback
   and for the existing (large-encoder) checkpoints.
4. **No retrieval-quality gate** (small-vs-large recall) for now — the user opted
   out; quality is observed after the eventual re-train + eval.

## Components & Changes (done now)

| # | Change | Where |
|---|--------|-------|
| 1 | Read encoder from `KG_ENCODER_MODEL` (default large) — add `import os`; replace the hardcoded `SentenceTransformer('abhinand/MedEmbed-large-v0.1', ...)` at line 65. | `baseline/scripts/serve/retrieval_tool.py` (1 line + import; back-compat) |
| 2 | **Rebuild KG** with the small encoder into `artifacts/kg_small/` (384-d FAISS indices + `medical_hg.json` + id maps). | run via `scripts.build_kg.run_pipeline --data-dir <abs artifacts/kg_small> --embed-model abhinand/MedEmbed-small-v0.1 --embed-device cpu` |
| 3 | `RETRIEVAL_CONTRACT_VERSION` → `"v1-medembed-small"`; `KGRetrieval` default `KG_ENCODER_MODEL` → `abhinand/MedEmbed-small-v0.1`. | `src/medical_qa_platform/retrieval/contract.py`, `kg_backend.py` |
| 4 | Parametrize the golden generator's encoder (add `--embed-model`, default large; record it in the manifest), then **regenerate** `tests/retrieval/golden/retrieve_v1_golden.json` from baseline `retrieve_v1` with `KG_ENCODER_MODEL=small` + `--data-dir artifacts/kg_small`. | `scripts/gen_retrieval_golden.py`, golden fixture |
| 5 | `params.yaml` `full`/`smoke_full` `build_kg.embed_model` → small. Wire `KG_ENCODER_MODEL` through the retrieval Helm chart (`values.env.kgEncoderModel` → deployment env). | `params.yaml`, `deploy/helm/retrieval/{values.yaml,templates/deployment.yaml}` |

The SP1 `ranker.py` is unchanged. The SP1 L1 test reads the regenerated golden
unchanged in shape (new data). The L2 test runs `KGRetrieval` with the small
encoder against `artifacts/kg_small`.

## Verification (controller-run; no full train)

- **Build:** confirm `artifacts/kg_small` indices load and the embedding
  dimension is what MedEmbed-small produces (record it; expected ~384).
- **L1 (CI, deterministic):** the regenerated golden (from baseline `retrieve_v1`
  on the small encoder) replays through `ranker.fuse_candidates` and matches.
- **L2 (opt-in, real):** full `KGRetrieval` (`KG_ENCODER_MODEL=small`,
  `KG_DATA_DIR=artifacts/kg_small`, CPU) reproduces the new golden, 8/8 queries.
- **Suite:** `make test` green; coverage gate holds; Helm render tests show the
  new `KG_ENCODER_MODEL` env.

## Out of Scope / Deferred

- **3-stage re-train** on the new encoder/KG — operator-run via the SP3 pipeline
  (`params.yaml` `build_kg.embed_model` is already set to small; run when GPU is
  free).
- **Retrieval-quality eval** (small vs large) — opted out for now.
- **SP4 contract-surfacing** (api `/version`, contract in `/predict` response) —
  remains a separate later item; only the encoder-config wiring is folded here.

## Risks / Open Questions

- **Dimension assumption:** if MedEmbed-small is not 384-d, nothing in code
  breaks (faiss + encoder are dim-agnostic), but the manifest should record the
  real dim; the index is dim-specific and must be rebuilt with the same encoder
  that serves.
- **Model download:** MedEmbed-small may need a one-time HF download (CPU path,
  no GPU needed).
- **Retrieval quality may drop** vs large (not measured by choice); will surface
  at post-retrain eval. Rollback path: `baseline/data` (large KG) + revert the
  contract version + golden.
- **Baseline edit must stay back-compatible** (default large) so existing
  large-encoder runs/checkpoints are unaffected.
- **GPU contention:** unrelated to SP2-now (KG build is CPU); only matters for
  the deferred re-train.
