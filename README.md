---
title: Shop SKU Manager
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Shop SKU Manager - OpenEnv Hackathon

AI agent that learns to optimize retail inventory management decisions.

## Setup

```bash
uv sync
source .venv/bin/activate
```

## Running

### Step 1: Start Ollama

```bash
ollama serve
```

Make sure you have a model pulled (e.g., `ollama pull mistral`).

### Step 2: Start the environment server

```bash
PYTHONPATH=src:envs SHOP_DIFFICULTY=easy \
  uv run uvicorn envs.shop_sku_manager.server.app:app --reload
```

### Step 3: Run the baseline agent

```bash
uv run python examples/shop_sku_baseline_ollama.py
```
