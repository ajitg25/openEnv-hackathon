FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN uv sync --no-install-project --no-editable || true

COPY . .
RUN uv sync --no-editable

RUN chown -R appuser:appuser /app

USER appuser

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/envs:/app:$PYTHONPATH"
ENV AMBULANCE_DIFFICULTY="easy"

EXPOSE 7860

# Showcase server: serves frontend + env API on one port
# v3: serve frontend at root /
CMD ["python", "showcase.py"]
