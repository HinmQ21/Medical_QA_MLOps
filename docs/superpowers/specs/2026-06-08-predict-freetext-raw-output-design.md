# Design: Free-text `/predict` request + raw model output in the response/UI

**Date:** 2026-06-08
**Status:** Approved (brainstorming) — pending spec review before planning

## Problem

Two limitations of the demo serving path:

1. **No raw output anywhere.** `/predict` runs `raw = backend.generate(...)`, then
   `answer = parse_answer(raw, valid_letters=set(req.options))` and **discards `raw`**
   ([`api/app.py:76-77`](../../../src/medical_qa_platform/api/app.py#L76-L77)).
   `PredictResponse` has no raw field, so neither the API nor the Streamlit UI can
   show the model's reasoning — only the single parsed letter. Since GBNF/constrained
   decoding was dropped, the model often answers in prose, so the parsed letter is
   frequently `null` and the user sees nothing useful.
2. **Two separate input boxes.** The UI ([`app/streamlit_app.py`](../../../app/streamlit_app.py))
   has a question `text_area` plus **four separate** A/B/C/D `text_input`s. The user
   wants a single combined box.

## Goal

- Expose the raw model generation through `/predict` and render it in the UI.
- Collapse the question + options into a **single free-text input** end-to-end: the
  `/predict` request takes one `question` field (options inline), no structured `options`.

## Decisions (from brainstorming)

- **Input contract = free-text.** Drop structured `options` from `PredictRequest`; the
  single `question` string carries the options inline. (Chosen over UI-side option
  parsing; the user accepts losing letter-validation against an option set and the
  per-option highlight.)
- **Keep best-effort answer parsing.** `/predict` still parses `<answer>` → letter
  (now `valid_letters=None`) and returns `answer` (nullable) **alongside** `raw_output`,
  so the structured answer letter remains available for eyeballing/accuracy.
- **Approach A — remove `options` entirely** (not keep it optional/deprecated). Pydantic
  v2 ignores extra fields by default, so a stray `options` from an old caller causes no
  422; we update our own smoke payload.

## Design

### 1. Contract — `api/schemas.py`

- `PredictRequest`: **remove** the `options` field and the `_validate_options` validator.
  Keep `question: str` (now free-text containing options inline) and `_question_not_blank`.
- `PredictResponse`: **add** `raw_output: str` — the full text the model generated.
  `answer: str | None` and all other fields unchanged.

### 2. Server — `api/`

- `prompt.build_prompt(question, evidence)` — **drop the `options` parameter**. Body
  becomes `Evidence:` (if any) + `Question: <text>` only; the options are already inside
  `<text>`. **`SYSTEM_PROMPT` is unchanged** (stays verbatim-synced with the training
  prompt — `test_system_prompt_sync` must still pass).
- `app.py` `/predict`: `messages = build_prompt(req.question, evidence)`;
  `answer = parse_answer(raw)` (no `valid_letters`); set `raw_output=raw` on the response.
- `parser.py` — **no change** (`valid_letters` is already optional; calling without it
  yields best-effort single-letter extraction).
- `drift/collector.py` — `record()` no longer has `request.options`. New row drops
  `n_options` and `mean_option_len`; keeps `q_token_len` (now counts the full combined
  text), `answer`, `no_result`, `latency_ms`. Signature `record(request, response,
  n_evidence)` unchanged.

### 3. UI / client — `app/`

- `client.build_payload(question)` — drop the `options` parameter and its validation;
  return `{"question": question}` after the non-blank check.
- `client.PredictResult` — add `raw_output: str`; `predict()` reads
  `data.get("raw_output", "")`.
- `streamlit_app.py`:
  - Replace the question `text_area` + four A/B/C/D `text_input`s with **one
    `text_area`** labelled "Câu hỏi (kèm phương án)".
  - `PRESETS` become single combined strings (question + `A) … B) … C) … D) …`).
  - Remove the per-option highlight loop. Render order: "Đáp án: **X**" if parsed
    (warning if `null`) → **expander "🧠 Phản hồi thô của mô hình"** (`expanded=False`)
    showing `raw_output` → existing evidence expander → metadata caption.

### 4. Smoke + tests

- `scripts/cloud/smoke_cloud.sh`: change `PAYLOAD` to free-text — embed the options in
  `question`, remove the `options` field. (`/predict` still returns `"answer"`; the grep
  assertion is unchanged.)
- Update tests: `tests/api/test_schemas.py`, `tests/api/test_app.py`,
  `tests/api/test_prompt.py`, `tests/app/test_ui_client.py`, `tests/drift/test_collector.py`,
  `tests/cloud/test_smoke_cloud.py`.
- **Unaffected** (keyword-match only, no change): `tests/api/test_parser.py`,
  `tests/api/test_system_prompt_sync.py`, `tests/mlops/*`,
  `tests/retrieval/golden/retrieve_v1_golden.json`,
  `tests/deploy/test_cloud_workflows.py`, `tests/deploy/test_kserve_chart.py`.

## Non-goals

- nginx `map_hash_bucket_size` fix (tracked separately under the in-cluster-KServe
  automation, "Path B").
- GBNF / constrained decoding.
- KServe / GKE-Standard provisioning automation.
- Changing `RETRIEVAL_CONTRACT_VERSION` (it versions the retrieval tool contract, not the
  predict request schema).

## Risks / notes

- `q_token_len` in the drift row now measures the combined question+options text, so its
  absolute values shift after this change; documented as expected, not a regression.
- With `valid_letters=None`, a model that emits a word in `<answer>` (e.g.
  `<answer>Metformin</answer>`) still yields `answer=null` (the regex matches a single
  letter only) — the full text is visible via `raw_output`.
- Old callers sending `{question, options}` are not rejected (extra field ignored), but
  their `question` will lack inline options unless updated — only our own smoke is updated.
