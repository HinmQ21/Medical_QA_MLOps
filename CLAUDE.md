# Project Snapshot

- **Dự án:** `mlops-platform/` — nền tảng **MLOps serving + pipeline** cho hệ Medical QA. Đây là project cá nhân, **tập trung kỹ thuật MLOps** (không phải nghiên cứu).
- **Quan hệ với `baseline/`:** `baseline/` là **khóa luận đã đóng băng** (research: KG + GRPO/GDPO, best MedQA Llama GDPO = 60.09%). Nó chỉ là **read-only reference + input cho pipeline**. `mlops-platform/` **KHÔNG được import `baseline/`** — có guard test ép buộc điều này.
- **Bản chất:** đóng gói phần training/eval của khóa luận thành 1 platform có: FastAPI inference API (agentic tool-call loop), KG retrieval service, DVC + MLflow pipeline, Helm/Docker/KServe, CI/CD keyless lên GKE, Streamlit demo UI, drift + Prometheus observability.
- **Git:** `mlops-platform/` là **repo riêng** (origin `https://github.com/HinmQ21/Medical_QA_MLOps.git`, branch `main`). `baseline/` là repo khác.
- **Package:** `medical_qa_platform` (src layout), Python ≥ 3.12, quản lý qua `uv` trong `.venv`.
- **Hardware serving thật:** model 3B chạy trên DGX-Spark GB10 (vLLM) expose qua Cloudflare Tunnel; hoặc Qwen2.5-1.5B llama.cpp in-cluster trên GKE Standard.
- **Trạng thái:** mọi sub-project (retrieval parity SP1–SP4, slim Plan 4 GKE demo, DGX vLLM backend, KServe llama.cpp, Streamlit UI, /predict free-text, agentic loop) đã merge `main` local; test suite xanh (~95% coverage). Live-validated end-to-end trên GKE nhiều lần.

# What Matters Most

1. **Working directory là `mlops-platform/`** — KHÔNG phải `baseline/`. Mọi lệnh `make`, `pytest`, deploy đều chạy từ đây:
   ```bash
   cd /home/vcsai/minhlbq/mlops-platform
   ```
2. **Boundary cứng: `src/medical_qa_platform/` tự chứa, không đụng `baseline/`.** Guard test `tests/retrieval/test_kg_backend_self_contained.py` cấm token `baseline` / `scripts.serve` / `MedicalKnowledgeTool` / `/home/vcsai/minhlbq` trong `src/`. Khi cần logic từ baseline (ranker, system prompt) → **port sang code thuần**, đừng import.
3. **Một venv duy nhất: `.venv` (python3.12 + uv).** Extras tách theo nhu cầu:
   - `[dev]` — pytest, cov, PyYAML (mặc định cho test)
   - `[pipeline]` — `dvc[gs]`, mlflow (chạy DVC/MLflow)
   - `[demo]` — streamlit (UI)
   - `[runtime]` — faiss-cpu, sentence-transformers (CHỈ container retrieval; không bao giờ trong unit test)
4. **Hai model backend (`MODEL_BACKEND`):**
   - `mock` (mặc định) — `MockBackend`, dùng cho CI/demo plumbing offline
   - `llm` (canonical; **`vllm` là alias back-compat**) — `LLMBackend`, client OpenAI `/v1` generic. Trỏ tới: (a) DGX vLLM qua Cloudflare Tunnel (3B GDPO thật), hoặc (b) KServe llama.cpp in-cluster (Qwen2.5-1.5B)
   - `kserve` và `runpod` backend **đã bị xóa** — `get_backend(...)` raise. KServe giờ là 1 InferenceService llama.cpp, **truy cập qua backend `llm`** chứ không có code backend riêng.
5. **`/predict` = agentic tool-call loop** (model-driven, native OpenAI `tools=`), KHÔNG phải one-shot RAG. Mô hình tự quyết khi nào gọi `search_medical_knowledge`; loop chạy tool → feed kết quả → lặp tới khi trả lời hoặc hết `MAX_TOOL_ITERATIONS` (mặc định 2). Mirror rollout của training. Response trả `answer` (parse best-effort), `raw_output`, `evidence`, `trace[]` (per-turn), `contract_version`, `trace_id`, `latency_ms`.
6. **Retrieval parity là bất biến.** Ranker `retrieve_v1` (8-term dual-retrieval fusion) được port nguyên vào `retrieval/ranker.py`; encoder = **`abhinand/MedEmbed-small-v0.1` (384-d)**; contract version = **`v1-medembed-small`** (`retrieval/contract.py`). Đổi ranker/encoder ⇒ **regenerate golden** (`scripts/gen_retrieval_golden.py`) + bump contract version + (nếu đổi encoder) **re-train**. Golden parity test: L1 (CI, replay FAISS hits) + L2 (opt-in, full backend).
7. **Phương pháp làm việc: brainstorm → spec → plan → subagent-driven.** Mọi feature lớn có spec + plan trong `docs/superpowers/{specs,plans}/<date>-*.md` (và `docs/{specs,plans}/` cho các plan cũ). Đọc spec/plan tương ứng trước khi sửa feature. Runbook vận hành ở `docs/runbooks/` và `docs/cloud-setup.md`.
8. **`artifacts/`, `mlruns/`, `.venv/`, `.tools/` là generated** — gitignored, không sửa tay. `baseline/` base-models, datasets, SFT traces là **read-only input** cho full pipeline.
9. **Test trước khi claim done:** `make test` (pytest + coverage, `fail_under=80`, thực tế ~95%). Một số test SKIP ngoài CI (cần extra `[runtime]`/`[demo]`) — verify thật trong venv có extra đó.
10. **CI demo dùng `mock`; flip sang `llm` khi muốn model thật.** Push `main` → `ci.yml` build 4 image lên GHCR; deploy lên GKE qua workflow. Auto Deploy chỉ roll khi cluster đang up.

# Repository Map

```
/home/vcsai/minhlbq/                  ← workspace root (anchor cho các path tuyệt đối bên dưới)
├── baseline/                         ← KHÓA LUẬN ĐÓNG BĂNG (read-only reference; repo riêng)
├── serving/                          ← gpt-oss-120b weights tải về (teacher serving thử nghiệm; ngoài phạm vi platform)
├── docs/                             ← tài liệu khóa luận (TONG_HOP_DU_AN.md, benchmark summaries...)
└── mlops-platform/                   ← ★ WORKING DIRECTORY + CLAUDE.md ở đây (repo riêng)
    ├── src/medical_qa_platform/      ← runtime package (cài qua pip -e)
    │   ├── config.py                       ← Settings.from_env (MODEL_BACKEND, TOP_K, MAX_TOKENS, MAX_TOOL_ITERATIONS...)
    │   ├── api/
    │   │   ├── app.py                       ← ★ FastAPI: /health /ready /predict /metrics /version
    │   │   ├── schemas.py                   ← PredictRequest{question}, PredictResponse{answer,raw_output,evidence,trace,...}, Turn
    │   │   ├── prompt.py                    ← build_prompt + SYSTEM_PROMPT (sync với training, có guard test)
    │   │   └── parser.py                    ← parse_answer(<answer>X</answer>), best-effort
    │   ├── inference/
    │   │   ├── base.py                      ← ModelBackend.chat(messages, tools, ...) -> ChatTurn; generate() wrapper
    │   │   ├── llm_backend.py               ← ★ LLMBackend (OpenAI /v1 client), name="llm"
    │   │   ├── mock_backend.py              ← MockBackend (default)
    │   │   ├── agent.py                     ← ★ run_agentic_loop(...) -> LoopResult(trace, evidence, ...)
    │   │   └── tools.py                     ← MEDICAL_TOOL_DEF / SEARCH_TOOL_NAME (mirror training schema)
    │   ├── retrieval/
    │   │   ├── ranker.py                    ← ★ retrieve_v1 thuần (8-term fusion) port từ baseline
    │   │   ├── kg_backend.py                ← KGRetrieval (FAISS + encoder), delegate vào ranker
    │   │   ├── contract.py                  ← ★ format_evidence + RETRIEVAL_CONTRACT_VERSION="v1-medembed-small"
    │   │   ├── service.py                   ← FastAPI retrieval service (process #2)
    │   │   ├── client.py                    ← RetrievalClient (api → retrieval HTTP)
    │   │   ├── backends.py                  ← SupportsSearch protocol
    │   │   └── device.py                    ← chọn CPU/GPU cho encoder
    │   ├── drift/collector.py               ← DriftCollector (ghi feature drift, best-effort)
    │   └── observability/                   ← logging.py, metrics.py (Prometheus)
    ├── mlops/
    │   ├── pipelines/
    │   │   ├── build_smoke_kg.py, build_demo_kg.py    ← KG nhỏ cho smoke / GKE demo
    │   │   ├── eval_smoke.py                          ← eval offline (mock)
    │   │   ├── run_full.py, full_stages.py, full_config.py, stage_runner.py  ← ★ orchestrate baseline training qua subprocess
    │   │   └── profiles.py                            ← load profile từ params.yaml
    │   ├── mlflow_register.py               ← đăng ký model vào MLflow registry
    │   └── smoke_data/medical_mcq.jsonl
    ├── deploy/helm/{api,retrieval,nginx,kserve,ui}/   ← 5 Helm chart
    ├── docker/{api,retrieval,pipeline-init,ui}.Dockerfile   ← 4 image (build --platform linux/amd64)
    ├── app/{client.py, streamlit_app.py}   ← Streamlit demo UI (client thuần + render)
    ├── scripts/
    │   ├── cloud/*.sh                       ← provision/deploy/secrets/smoke/teardown + demo_up_llm + install_kserve
    │   ├── gen_retrieval_golden.py          ← regenerate golden parity (chạy ở venv baseline)
    │   └── install_deploy_tools.py          ← tải helm/kubectl vào .tools/bin
    ├── .github/workflows/{ci,demo-up,demo-down,deploy}.yml
    ├── tests/{api,inference,retrieval,drift,observability,mlops,deploy,cloud,app,serving}/  ← 71 file test
    ├── docs/{specs,plans,runbooks,cloud-setup.md, superpowers/{specs,plans}}/
    ├── params.yaml                          ← profiles: smoke / full / smoke_full
    ├── dvc.yaml + dvc.lock                  ← stage lineage (smoke + demo_kg + full_*)
    ├── pyproject.toml                       ← deps + extras [dev|pipeline|demo|runtime]
    ├── Makefile                             ← entrypoint mọi tác vụ
    ├── artifacts/, mlruns/                  ← GENERATED (gitignored)
    └── .venv/, .tools/                      ← venv + helm/kubectl local
```

**Đọc trước khi sửa serving:** `inference/agent.py` + `inference/base.py` + `api/app.py` + `inference/tools.py`.
**Đọc trước khi sửa retrieval:** `retrieval/ranker.py` + `retrieval/contract.py` + `retrieval/kg_backend.py` + test parity.

# Architecture

```
                         ┌─────────────── GKE (demo) ───────────────┐
Browser ── UI(Streamlit :8501, own LB) ──x-api-key──> nginx gateway ─┐
                                                                     │ (x-api-key server-side)
                                            ┌────────────────────────▼─────────────┐
                                            │  api (FastAPI)  — /predict            │
                                            │  agentic tool-call loop               │
                                            └───┬───────────────────────────────┬───┘
                            tool: search_medical_knowledge                  ModelBackend
                                                │                               │
                                  retrieval service (FAISS + MedEmbed-small)    │
                                  ranker retrieve_v1 → contract.format_evidence │
                                                                                │
                              MODEL_BACKEND=mock ── MockBackend                 │
                              MODEL_BACKEND=llm  ── LLMBackend (OpenAI /v1) ────┤
                                                       ├── DGX vLLM (3B GDPO) qua Cloudflare Tunnel
                                                       └── KServe llama.cpp (Qwen2.5-1.5B) in-cluster

Offline:  params.yaml ─▶ mlops/pipelines ─▶ DVC stages ─▶ artifacts/  ─▶ MLflow registry
          full pipeline gọi baseline training bằng subprocess (KHÔNG import baseline)
```

## /predict — agentic loop chi tiết

- Request: `{ "question": "<câu hỏi kèm A) B) C) D) inline>" }` (đã bỏ field `options` có cấu trúc; caller cũ gửi `options` không bị 422, chỉ bị bỏ qua).
- Loop (`run_agentic_loop`): vòng `0..max_iterations-1` offer `tools=[MEDICAL_TOOL_DEF]`; vòng cuối (`= max_iterations`) bỏ tools để ép model trả lời text.
- Tool call → `RetrievalClient.search(query, top_k)` → evidence string nối vào `messages` (role `tool`) + ghi vào `trace`.
- Response: `answer` (letter parse best-effort, có thể `null`), `raw_output` (text thô đầy đủ), `evidence[]`, `trace[]` (per-turn để UI hiển thị), `backend`, `model_version`, `contract_version`, `latency_ms`, `trace_id`.

## Backend `llm` (OpenAI /v1) — hai đích

| Đích | Khi nào | Cấu hình |
|------|---------|----------|
| **DGX vLLM + Cloudflare Tunnel** | model 3B GDPO thật | `LLM_BASE_URL=https://llm.<domain>/v1`, `LLM_MODEL=medical-qa-llama-gdpo`, `LLM_API_KEY=...`. Runbook `docs/runbooks/dgx-vllm-cloudflare.md` |
| **KServe llama.cpp in-cluster** | demo GKE Standard, không cần DGX | `LLM_BASE_URL=http://medical-qa-kserve-predictor.<ns>.svc.cluster.local/v1`, `LLM_MODEL=qwen2.5-1.5b-instruct`, `LLM_API_KEY` unset |

## Env vars chính (xem `config.py`)

`MODEL_BACKEND` (mock|llm; vllm alias) · `RETRIEVAL_URL` · `MODEL_VERSION` · `TOP_K` (5) · `MAX_TOKENS` (config default 512 — **deploy set 2048** để không cắt cụt `<answer>`) · `MAX_TOOL_ITERATIONS` (2) · `DRIFT_LOG_PATH` (deploy trỏ `/tmp`, non-root) · `LLM_BASE_URL` (**phải kết thúc `/v1`**) · `LLM_MODEL` · `LLM_API_KEY` · `KG_DATA_DIR` · `KG_ENCODER_MODEL`.

# Commands

```bash
cd /home/vcsai/minhlbq/mlops-platform   # luôn từ đây

# ── Setup ──
make install            # .venv + uv pip install -e ".[dev]"
make install-pipeline   # thêm extra [pipeline] (dvc[gs], mlflow)
make install-deploy-tools   # tải helm + kubectl vào .tools/bin

# ── Test (luôn chạy trước khi claim done) ──
make test               # pytest + cov, fail_under=80 (~95% thực tế)
# test [runtime]/[demo] bị SKIP nếu thiếu extra → cài extra rồi chạy lại để verify thật

# ── Dev API local ──
MODEL_BACKEND=mock .venv/bin/uvicorn medical_qa_platform.api.app:create_app --factory --reload
# trỏ model thật:
MODEL_BACKEND=llm LLM_BASE_URL=https://llm.<domain>/v1 LLM_MODEL=... LLM_API_KEY=... \
  .venv/bin/uvicorn medical_qa_platform.api.app:create_app --factory

# ── Streamlit demo UI ──
make demo-ui            # cài [demo] rồi streamlit run app/streamlit_app.py :8501
# (local: .venv/bin/pip install -U uv trước nếu .venv chưa có uv)

# ── Pipeline offline ──
make smoke-pipeline-local   # build_smoke_kg → eval_smoke → mlflow register (dry-run), không cần DVC
make smoke-pipeline         # cùng stage qua DVC (dvc repro)
make full-pipeline-dry-run  # in kế hoạch full pipeline (CI = structural)
make full-pipeline          # CHẠY THẬT: orchestrate baseline training (cần GPU GB10 free, vLLM)
make smoke-full             # full pipeline với caps tí hon (gate cho run thật)
make dvc-status

# ── Docker + Helm ──
make docker-build       # 4 image, --platform linux/amd64 (host aarch64 → cross-build)
make helm-lint          # lint 5 chart (api/retrieval/nginx/kserve/ui)
make helm-template      # render 5 chart
make helm-dry-run       # kubectl apply --dry-run cho chart resource chuẩn

# ── Cloud / CI-CD (GKE) ──
# One-time bootstrap (durable):
make cloud-gcs-dvc            # GCS DVC remote
make cloud-workload-identity  # WI cho retrieval pull KG
make cloud-github-oidc        # keyless GitHub→GCP (WIF) + GSA quyền
# Demo Up/Down chạy qua GitHub workflows; hoặc local:
make cloud-provision          # GKE Autopilot (mock demo)
make cloud-secrets            # tạo nginx api-key + LLM key secret
make cloud-deploy             # helm upgrade --install các chart
make cloud-smoke              # smoke nginx→api→retrieval→model
make cloud-teardown           # xóa cluster + LB (giữ bucket)
# Demo GKE Standard + llama.cpp in-cluster (1 lệnh):
GCP_PROJECT=... NGINX_API_KEY=$(openssl rand -hex 24) bash scripts/cloud/demo_up_llm.sh

# ── Retrieval golden parity (khi đổi ranker/encoder) ──
python scripts/gen_retrieval_golden.py --embed-model abhinand/MedEmbed-small-v0.1   # chạy ở venv baseline
# rồi bump RETRIEVAL_CONTRACT_VERSION + chạy lại parity test (L1 CI / L2 opt-in)
```

# Working Rules for Claude

- **Mặc định mọi việc trong `mlops-platform/`.** Chỉ đụng `baseline/` để **đọc** (reference) hoặc làm **input** cho full pipeline — không sửa, không import vào `src/`.
- **Giữ boundary self-contained:** trước khi thêm import trong `src/`, nhớ guard test cấm chuỗi `baseline`/`scripts.serve`/`MedicalKnowledgeTool`/`/home/vcsai/minhlbq`. Cần logic baseline → port code thuần. Đừng viết path `baseline` ngay cả trong docstring runtime.
- **Đổi system prompt phải sync** giữa serving (`api/prompt.py`) và training — `tests/api/test_system_prompt_sync.py` (hoặc tương đương) sẽ fail nếu lệch.
- **Đổi ranker hoặc encoder** ⇒ regenerate golden (`scripts/gen_retrieval_golden.py`) + bump `RETRIEVAL_CONTRACT_VERSION` + (đổi encoder) re-train 3 stage qua `make full-pipeline`.
- **Backend mới đi qua `get_backend`** (`inference/__init__.py`). Đừng tái sinh backend `kserve`/`runpod` — đã xóa có chủ đích; route model in-cluster qua backend `llm`.
- **`MAX_TOKENS` deploy phải ≥ 2048** (default 512 trong config sẽ cắt cụt `<think>…</think><answer>` → mất `<answer>` → answer null).
- **Mỗi feature lớn: brainstorm → spec → plan → subagent-driven**, lưu spec/plan vào `docs/superpowers/{specs,plans}/<date>-*.md`. Đọc spec/plan liên quan trước khi sửa.
- **Helm: chart và backend tách rời.** Chart `kserve` deploy llama.cpp thật; API nối qua backend `llm` + `LLM_BASE_URL`. `deploy.sh` chỉ cài chart kserve khi có CRD (Autopilot không có → skip non-fatal).
- **`nginx` configmap: giữ `map_hash_bucket_size 128`** (API key dài >40 ký tự sẽ crashloop nếu để mặc định 64). Rotate `NGINX_API_KEY` ⇒ `rollout restart` CẢ `medical-qa-nginx` LẪN `medical-qa-ui`.
- **Image redeploy phải dùng sha tag** (`IMAGE_TAG=<sha> bash scripts/cloud/deploy.sh`) vì chart `pullPolicy: IfNotPresent` — `:latest` không kéo code mới. Auto Deploy nhắm Autopilot `medical-qa`, KHÔNG tự cập nhật cluster thủ công `medical-qa-llm`.
- **GHCR package phải PUBLIC** (chart không có imagePullSecret) — private → ImagePullBackOff.
- **Thêm experiment/pipeline mới:** thêm profile mới trong `params.yaml` (đừng overwrite `smoke`/`full`), thêm stage DVC tương ứng.
- **Không commit/sửa tay** `artifacts/`, `mlruns/`, `dvc.lock` (trừ khi `dvc repro`/`dvc push` hợp lệ), file trong `.venv`/`.tools`.
- **Đừng quote 60.09% MedQA cho serving path** — số đó là agentic loop trong eval khóa luận; serving (single-shot hoặc 1.5B in-cluster) phải đo riêng.

# Known Risks / Open Questions

- **Push state:** nhiều merge ở `main` là **local, chưa push** origin (xem từng feature). Kiểm `git log origin/main..HEAD` trước khi giả định CI đã chạy.
- **Auto Deploy ≠ cluster thủ công:** `deploy.yml` chỉ roll Autopilot regional `medical-qa`. Cluster `medical-qa-llm` (manual, GKE Standard + llama.cpp) phải deploy tay bằng sha tag.
- **KServe RawDeployment không scale-to-zero** (`minReplicas: 1` không KEDA). llama.cpp CPU ~14–20 tok/s ở 4 vCPU → latency cao; `/predict` timeout UI để 120s. `emptyDir` → re-download GGUF ~1.1GB mỗi restart (cân nhắc PVC).
- **Cloudflare free tier ~100s origin timeout (524)** — ổn với ≤2048 token, dài hơn sẽ đứt.
- **`MedEmbed-small` KG hiện gói cả feature-rich (~158.7K hedges)** — không cô lập được hiệu ứng encoder; muốn so v0+small cần rebuild `--skip-features`. Re-train 3 stage trên KG small này **chưa chạy** (operator/GPU).
- **Drift `no_result` gộp "tool không gọi" với "tool gọi nhưng rỗng"** (out of scope khi làm agentic loop); `tool_call_count`/`trace_id` chưa lên telemetry.
- **Test SKIP ngoài CI:** test cần `[runtime]` (faiss/sentence-transformers) hoặc `[demo]` (streamlit) bị `importorskip` → CI không cover. Verify thật trong venv có extra. `streamlit run` có sys.path trap (AppTest dưới pytest che mất) — có regression test riêng.
- **Tool path Qwen-only:** agentic loop dùng native OpenAI function-calling; chưa có nhánh tool cho Llama family.

# Suggested Skills to Create

| Skill | Khi nào dùng | Nội dung nên đưa vào |
|-------|-------------|---------------------|
| `cloud-demo-up` | Bật demo GKE (mock / llm / in-cluster) | Bootstrap WIF/OIDC, `demo_up_llm.sh` vs Demo Up workflow, gotchas (GHCR public, map_hash 128, sha tag), teardown nhắc billing |
| `full-pipeline-run` | Chạy `make full-pipeline` training thật | Cần GPU GB10 free + vLLM, profile `full`, baseline_root read-only, parse eval metrics nested, `smoke-full` gate trước |
| `retrieval-golden` | Đổi ranker/encoder | Quy trình regen golden ở venv baseline, bump contract version, L1/L2 parity, khi nào phải re-train |
| `serving-backend` | Thêm/đổi model backend | `get_backend`, LLMBackend OpenAI /v1, LLM_BASE_URL phải `/v1`, MAX_TOKENS ≥2048, đừng tái sinh kserve/runpod |
| `helm-deploy` | Deploy/sửa chart | 5 chart, chart↔backend tách rời, sha tag redeploy, rotate api-key restart cả nginx+ui, KServe CRD check |
```

