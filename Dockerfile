FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.12 python3.12-venv python3-pip git curl && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python deps (cached layer)
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-install-project --no-editable || true

# Training deps (GPU-only, not in pyproject.toml)
RUN uv pip install --system \
    "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git" \
    trl peft accelerate nest_asyncio \
    "openenv-core>=0.2.2" --no-deps \
    pydantic fastapi uvicorn websockets httpx matplotlib numpy

COPY . .
RUN uv sync --no-editable

RUN mkdir -p /app/plots && chown -R appuser:appuser /app

USER appuser

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/envs:/app:$PYTHONPATH"
ENV AMBULANCE_DIFFICULTY="easy"

EXPOSE 7860

# Training runs at container start, then switches to serving
# If plots already exist (persistent storage), skip training
CMD ["sh", "-c", "\
    if [ ! -f /app/plots/results.json ]; then \
        echo '=== TRAINING MODE ===' && \
        python3 /app/train.py; \
    else \
        echo '=== Training already done, skipping ==='; \
    fi && \
    echo '=== SERVING MODE ===' && \
    AMBULANCE_DIFFICULTY=easy uvicorn ambulance_env.server.app:app \
        --host 0.0.0.0 --port 7860 --app-dir /app/envs \
        --ws-ping-interval 300 --ws-ping-timeout 300 \
"]
