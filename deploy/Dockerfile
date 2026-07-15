# Cat-Agent on-prem runtime image (air-gap friendly defaults)
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CAT_AGENT_DEFAULT_WORKSPACE=/data/workspace \
    CAT_AGENT_OFFLINE=1 \
    CAT_AGENT_AUDIT=1 \
    CAT_AGENT_ENCRYPT_AT_REST=1 \
    CAT_AGENT_PII_REDACT=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/cat-agent

COPY pyproject.toml README.md ./
COPY cat_agent ./cat_agent
COPY native ./native

RUN pip install '.[rag,wasm-bundled,otel,pii]' \
    && mkdir -p /data/workspace /data/models /data/audit

VOLUME ["/data/workspace", "/data/models", "/data/audit"]

COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["offline-check"]
