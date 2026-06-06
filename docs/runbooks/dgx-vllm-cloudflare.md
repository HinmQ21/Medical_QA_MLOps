# Runbook: DGX-Spark vLLM behind Cloudflare Tunnel

Self-host the trained model on the DGX-Spark and expose it to the GKE API as the
`vllm` backend. Two long-lived processes run on the DGX under systemd: the vLLM
OpenAI server (loopback only) and `cloudflared` (named tunnel).

## Prerequisites
- Merged checkpoint on disk: `baseline/outputs/gdpo_llama32_3b_stage2_v6_1_vllm_merged`
- `vllm_venv312` present (see baseline/CLAUDE.md)
- A domain added to Cloudflare (free plan is fine)
- `cloudflared` installed on the DGX

## 1. vLLM OpenAI server (loopback)

Pick a strong key and export it once for the unit:
```bash
echo "DGX_LLM_KEY=$(openssl rand -hex 24)" | sudo tee /etc/medqa-llm.env
```

`/etc/systemd/system/medqa-vllm.service`:
```ini
[Unit]
Description=medqa vLLM OpenAI server
After=network-online.target

[Service]
User=vcsai
EnvironmentFile=/etc/medqa-llm.env
WorkingDirectory=/home/vcsai/minhlbq/baseline
ExecStart=/home/vcsai/minhlbq/baseline/vllm_venv312/bin/python -m vllm.entrypoints.openai.api_server \
  --model outputs/gdpo_llama32_3b_stage2_v6_1_vllm_merged \
  --served-model-name medical-qa-llama-gdpo \
  --host 127.0.0.1 --port 8001 \
  --api-key ${DGX_LLM_KEY} \
  --gpu-memory-utilization 0.6 --max-model-len 4096
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Note: `--served-model-name medical-qa-llama-gdpo` MUST equal the GKE `LLM_MODEL`.

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now medqa-vllm
curl -s -H "Authorization: Bearer $(grep -oP '(?<=DGX_LLM_KEY=).*' /etc/medqa-llm.env)" \
  http://127.0.0.1:8001/v1/models   # expect 200 + medical-qa-llama-gdpo
```

## 2. Cloudflare named tunnel

```bash
cloudflared tunnel login
cloudflared tunnel create medqa-llm          # writes ~/.cloudflared/<UUID>.json
cloudflared tunnel route dns medqa-llm llm.<your-domain>
```

`~/.cloudflared/config.yml`:
```yaml
tunnel: medqa-llm
credentials-file: /home/vcsai/.cloudflared/<UUID>.json
ingress:
  - hostname: llm.<your-domain>
    service: http://127.0.0.1:8001
  - service: http_status:404
```

`/etc/systemd/system/medqa-cloudflared.service`:
```ini
[Unit]
Description=medqa cloudflared tunnel
After=network-online.target medqa-vllm.service

[Service]
User=vcsai
ExecStart=/usr/bin/cloudflared tunnel run medqa-llm
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now medqa-cloudflared
curl -s -H "Authorization: Bearer <DGX_LLM_KEY>" https://llm.<your-domain>/v1/models
```

## 3. Point GKE at the DGX

```bash
# create the in-cluster secret from the same key
export LLM_API_KEY="<DGX_LLM_KEY>"
bash scripts/cloud/create_secrets.sh        # also (re)creates the nginx key

# deploy the api on the vllm backend (note the /v1 suffix)
export MODEL_BACKEND=vllm
export LLM_BASE_URL="https://llm.<your-domain>/v1"
export LLM_MODEL="medical-qa-llama-gdpo"
bash scripts/cloud/deploy.sh
```

Verify: `kubectl -n <ns> rollout status deploy/medical-qa-api`; the pod becomes
Ready only when `/ready` -> `health_check()` -> `GET <base_url>/models` succeeds,
i.e. the tunnel and vLLM are both up.

## Gotchas
- `LLM_BASE_URL` MUST end in `/v1` (backend appends `/chat/completions`).
- Cloudflare free tier drops origin requests after ~100s (error 524); a 3B model
  at <=2048 tokens answers in seconds, so this is only a concern under overload.
- Keep `--api-key` on; without it the tunnel URL is an open inference endpoint.
- Accuracy here is single-shot RAG (platform pre-retrieves evidence), not the
  agentic tool loop used in training — do not quote the 60.09% MedQA number for
  this serving path; measure it separately.

## Driving it from the GitHub workflows (instead of manual deploy)

The `Demo Up (GKE)` and `Auto Deploy` workflows deploy the `vllm` backend by
default. One-time GitHub repo config (Settings → Secrets and variables → Actions):

- **Secret** `LLM_API_KEY` — the DGX vLLM `--api-key`.
- **Var** `LLM_BASE_URL` — e.g. `https://llm.<your-domain>/v1` (must end in `/v1`).
- **Var** `LLM_MODEL` — e.g. `medical-qa-llama-gdpo` (= `--served-model-name`).
- **Var** `MODEL_BACKEND` (optional) — set to `mock` to make `Auto Deploy` skip the
  real model; otherwise it defaults to `vllm`.

Then: run **Demo Up (GKE)** (leave `backend: vllm`) to provision + deploy the real
model. After that, every push to `main` that passes CI triggers **Auto Deploy**,
which re-applies the secrets and rolls the api on `vllm` — provided the DGX server
and Cloudflare Tunnel are up (otherwise the api pod stays NotReady and the run
fails loudly). To demo the plumbing with the DGX offline, run **Demo Up** with
`backend: mock`.
