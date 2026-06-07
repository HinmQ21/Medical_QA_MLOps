# Runbook: DGX-Spark vLLM behind a public tunnel

Self-host the trained model on the DGX-Spark and expose it to the GKE API as the
`vllm` backend. Two long-lived pieces run on the DGX: a vLLM OpenAI server
**container** (loopback only) and a public tunnel — **Tailscale Funnel (recommended,
no domain)** or a Cloudflare named tunnel.

## Prerequisites
- A merged checkpoint dir to serve, e.g. `baseline/outputs/<run>_merged`. If the
  fine-tuned checkpoint isn't on disk, mount the base model
  `baseline/models/Llama-3.2-3B-Instruct` first to validate the path end-to-end.
- Docker with the NVIDIA runtime (DGX-Spark has both) and a vLLM image —
  `vllm/vllm-openai:latest` or `nvcr.io/nvidia/vllm:<tag>-py3` (preferred on the
  GB10 / ARM64 box). Non-Docker alternative: `vllm_venv312` (see baseline/CLAUDE.md).
- A public tunnel for the loopback vLLM port (§2): **Tailscale** (no domain — recommended)
  or `cloudflared` + a Cloudflare domain. A quick tunnel works only for throwaway tests.

## 1. vLLM OpenAI server (Docker, loopback)

vLLM runs as a container bound to host loopback; `cloudflared` is the only thing that
exposes it. Docker is the recommended path on the GB10 (ARM64 + Blackwell) — the
image already ships the right CUDA kernels, so there's no venv/torch build to fight.

Pick a strong key and the model dir to mount (the merged fine-tuned checkpoint, or the
base model to test the plumbing first):
```bash
echo "DGX_LLM_KEY=$(openssl rand -hex 24)"                                    | sudo tee  /etc/medqa-llm.env
echo "MODEL_DIR=/home/vcsai/minhlbq/baseline/models/Llama-3.2-3B-Instruct"    | sudo tee -a /etc/medqa-llm.env
```

`/etc/systemd/system/medqa-vllm.service` (systemd owns the container lifecycle):
```ini
[Unit]
Description=medqa vLLM OpenAI server (Docker)
After=network-online.target docker.service
Requires=docker.service

[Service]
EnvironmentFile=/etc/medqa-llm.env
# --ipc=host is required by vLLM (shared memory); --rm + foreground so systemd owns it.
ExecStartPre=-/usr/bin/docker rm -f medqa-vllm
ExecStart=/usr/bin/docker run --rm --name medqa-vllm \
  --gpus all --ipc=host \
  -p 127.0.0.1:8001:8000 \
  -v ${MODEL_DIR}:/model:ro \
  -e VLLM_API_KEY=${DGX_LLM_KEY} \
  vllm/vllm-openai:latest \
  --model /model --served-model-name medical-qa-llama-gdpo \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.6 --max-model-len 4096
ExecStop=/usr/bin/docker stop medqa-vllm
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Notes:
- `--served-model-name medical-qa-llama-gdpo` MUST equal the GKE `LLM_MODEL` var.
- The container listens on `0.0.0.0:8000` *inside* the container; `-p 127.0.0.1:8001:8000`
  publishes it to host loopback only, so `cloudflared` (and nothing else) can reach it.
- NVIDIA NGC image instead? Its entrypoint is the nvidia wrapper, so pass the full
  command: replace the image + args with
  `nvcr.io/nvidia/vllm:26.01-py3 vllm serve /model --host 0.0.0.0 --port 8000 --served-model-name medical-qa-llama-gdpo --gpu-memory-utilization 0.6 --max-model-len 4096`.

<details><summary>Alternative: run without Docker (vllm_venv312)</summary>

```ini
[Service]
User=vcsai
EnvironmentFile=/etc/medqa-llm.env
WorkingDirectory=/home/vcsai/minhlbq/baseline
ExecStart=/home/vcsai/minhlbq/baseline/vllm_venv312/bin/python -m vllm.entrypoints.openai.api_server \
  --model outputs/<run>_merged \
  --served-model-name medical-qa-llama-gdpo \
  --host 127.0.0.1 --port 8001 \
  --api-key ${DGX_LLM_KEY} \
  --gpu-memory-utilization 0.6 --max-model-len 4096
Restart=always
RestartSec=5
```
</details>

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now medqa-vllm
curl -s -H "Authorization: Bearer $(grep -oP '(?<=DGX_LLM_KEY=).*' /etc/medqa-llm.env)" \
  http://127.0.0.1:8001/v1/models   # expect 200 + medical-qa-llama-gdpo
```

## 2. Expose the server to GKE (pick one tunnel)

The GKE API reaches the DGX over a tunnel; its URL becomes the GKE `LLM_BASE_URL`, so it
must be **stable**. Do NOT use a quick tunnel for a demo — it exits on the first
connection drop and its URL rotates on every restart. (Quick **test** only, no
login/domain: `cloudflared tunnel --url http://127.0.0.1:8001` → ephemeral
`https://<random>.trycloudflare.com`; set it as `LLM_BASE_URL` + `/v1` and expect it to
die within hours.)

### 2a. Tailscale Funnel (recommended — stable URL, no domain)

Public HTTPS at `https://<host>.<tailnet>.ts.net`, nothing to buy; `tailscaled`
auto-reconnects and the Funnel config (`--bg`) survives reboot — far more robust than a
quick tunnel. Set `LLM_BASE_URL` once and never touch it again.

```bash
curl -fsSL https://tailscale.com/install.sh | sh    # arm64-aware
sudo tailscale up                                   # open the printed auth URL to log in
# Admin console (login.tailscale.com/admin): DNS -> enable MagicDNS + HTTPS Certificates;
# Access Controls -> allow Funnel via nodeAttrs: {"target":["autogroup:member"],"attr":["funnel"]}
sudo tailscale funnel --bg 8001                     # public :443 -> http://127.0.0.1:8001
tailscale funnel status                             # prints https://<host>.<tailnet>.ts.net
tailscale status --json | python3 -c 'import sys,json;print(json.load(sys.stdin)["Self"]["DNSName"])'  # FQDN
```

Then `LLM_BASE_URL = https://<host>.<tailnet>.ts.net/v1` (no port, keep `/v1`). vLLM's
`--api-key` still gates the (public) Funnel endpoint, so auth is enforced over the internet.

### 2b. Cloudflare named tunnel (alternative — needs a Cloudflare domain)

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
