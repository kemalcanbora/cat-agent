# On-prem deployment package

Deploy Cat-Agent in an air-gapped network with encrypted storage, audit logging,
and offline guards enabled by default.

## Quick start (connected machine → transfer image)

```bash
# 1. Generate an encryption key
python - <<'PY'
import base64, secrets
print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())
PY

# 2. Configure environment
cp deploy/.env.example deploy/.env
# Edit deploy/.env and paste CAT_AGENT_ENCRYPTION_KEY

# 3. Build the image
docker compose -f deploy/docker-compose.yml build

# 4. Save image for offline transfer
docker save cat-agent:on-prem | gzip > cat-agent-on-prem.tar.gz

# 5. On the air-gapped host
docker load < cat-agent-on-prem.tar.gz
docker compose -f deploy/docker-compose.yml up
```

## Volumes

| Volume | Purpose |
|---|---|
| `workspace` | Doc caches, RAG indexes, agent memory (SQLite) |
| `models` | Local model weights / embeddings (mount your ONNX/GGUF files here) |
| `audit` | Tamper-evident audit JSONL |

## Pre-flight checks

```bash
docker compose -f deploy/docker-compose.yml run --rm cat-agent offline-check --strict
docker compose -f deploy/docker-compose.yml run --rm cat-agent encrypt-storage
```

## Python usage inside the container

```bash
docker compose -f deploy/docker-compose.yml run --rm cat-agent python -c "import cat_agent; print(cat_agent.__version__)"
```

Point your application at the mounted `/data/workspace` and configure an
on-prem OpenAI-compatible endpoint via `OPENAI_BASE_URL` in `deploy/.env`.
Set `CAT_AGENT_OFFLINE_ALLOW_HOSTS` to permit internal LLM gateways while
`CAT_AGENT_OFFLINE=1` blocks the public internet. Docker Compose and
Cat-Agent both read this `.env` file automatically.

## Kubernetes CronJob (scheduled reports)

Manifests under `deploy/k8s/` run the shared oneshot driver on a poll cadence
(`*/15 * * * *`). Per-user cadence is stored in `jobs.next_run_at`, not in the
CronJob schedule.

```bash
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/cronjob.yaml
```

Requires `pip install 'cat-agent[scheduler]'` in the image and a PVC named
`cat-agent-workspace` (or edit the volume claim). Replace Secret placeholders
before applying to a real cluster.

