#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Baseline inference script for Shop SKU Manager environment.

Uses Ollama (local LLM) to run an intelligent agent that learns inventory management.
Evaluates on EASY difficulty (for fast local testing).

Requirements:
    - Environment server running: PYTHONPATH=src:envs SHOP_DIFFICULTY=easy uv run uvicorn envs.shop_sku_manager.server.app:app
    - Ollama running locally on port 11434
    - A model pulled (e.g., ollama pull llama2 or ollama pull mistral)

Usage:
    # Terminal 1: Start environment server
    PYTHONPATH=src:envs SHOP_DIFFICULTY=easy uv run uvicorn envs.shop_sku_manager.server.app:app --host 127.0.0.1 --port 8000

    # Terminal 2: Start Ollama
    ollama serve

    # Terminal 3: Run baseline
    python examples/shop_sku_baseline_ollama.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "envs"))

from shop_sku_manager.client import ShopSKUManagerEnv
from shop_sku_manager.models import OrderAction

try:
    from openai import AsyncOpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)


async def baseline_agent():
    """
    Baseline agent using Ollama (local LLM).

    Agent reasons about:
    - Current inventory levels
    - Demand forecasts
    - Lead times
    - Budget constraints

    And makes order decisions to maximize profit.
    """

    # Check if Ollama is running
    print("\n" + "="*70)
    print("🛒 SHOP SKU MANAGER - BASELINE INFERENCE (OLLAMA)")
    print("="*70)
    print("\n⚙️  Checking Ollama connection...")

    # Create Ollama client (OpenAI-compatible endpoint)
    client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Dummy key for local Ollama
    )

    # Try to list available models
    try:
        response = await client.models.list()
        print(f"✅ Connected to Ollama!")
        available_models = [m.id for m in response.data] if hasattr(response, 'data') else []
        print(f"   Available models: {available_models}")
    except Exception as e:
        print(f"\n❌ Error: Could not connect to Ollama at http://localhost:11434")
        print(f"   Make sure Ollama is running:")
        print(f"   1. Download from https://ollama.ai")
        print(f"   2. Run: ollama serve")
        print(f"   3. In another terminal: ollama pull mistral (or llama2 or neural-chat)")
        print(f"\n   Error: {str(e)}\n")
        sys.exit(1)

    # Select model: prefer fast models, fallback to first available
    preferred_models = ["mistral", "neural-chat", "llama2", "orca-mini"]
    model_name = None
    for preferred in preferred_models:
        if preferred in available_models:
            model_name = preferred
            break

    if not model_name and available_models:
        model_name = available_models[0]

    if not model_name:
        print(f"\n❌ Error: No models available in Ollama")
        print(f"   Pull a model first: ollama pull mistral\n")
        sys.exit(1)

    print(f"   Using model: {model_name}")

    scores = []

    print("\n" + "="*70 + "\n")
    print("ℹ️  Testing EASY difficulty")
    print("   (Make sure server started with: SHOP_DIFFICULTY=easy)")
    print("="*70 + "\n")

    server_url = os.getenv("SERVER_URL", "https://ajitg25-openenv-hackathon.hf.space/")
    print(f"   Server: {server_url}\n")

    for episode in range(3):  # 3 episodes
        print(f"⚡ Episode {episode+1}/3...")

        # Create new client for this episode
        try:
            env = ShopSKUManagerEnv(base_url=server_url)
        except Exception as e:
            print(f"\n❌ Error: Could not connect to server at {server_url}")
            print(f"   Details: {str(e)}")
            print(f"   Make sure the server is running, or set SERVER_URL env var:")
            print(f"   SERVER_URL=https://ajitg25-openenv-hackathon.hf.space uv run python examples/shop_sku_baseline_ollama.py\n")
            sys.exit(1)

        try:
            result = await env.reset()
            episode_profit = 0.0
            day = 0
            print(f"   ✓ Reset OK, got {len(result.observation.inventory_levels)} SKUs")

            while not result.done:
                day += 1
                obs = result.observation

                # Build prompt for agent
                inventory_summary = ", ".join(
                    [f"{sku}: {int(qty)} units"
                     for sku, qty in obs.inventory_levels.items()]
                )

                forecast_summary = ", ".join(
                    [f"{sku}: {obs.demand_forecast[sku]:.1f}"
                     for sku in obs.inventory_levels.keys()]
                )

                lead_times_summary = ", ".join(
                    [f"{sku}: {obs.lead_times[sku]} days"
                     for sku in obs.inventory_levels.keys()]
                )

                prompt = f"""You are a shop inventory manager. Make ordering decisions to maximize profit.

Current Status (Day {day}):
- Inventory: {inventory_summary}
- Budget: ${obs.budget_remaining:.2f}
- Forecasted demand: {forecast_summary}
- Lead times: {lead_times_summary}

IMPORTANT: Respond ONLY with valid JSON. No other text.

Example valid responses:
{{"orders": {{"milk": 15, "bread": 10}}, "emergency": false}}
{{"orders": {{}}, "emergency": false}}

Your decision (JSON only):"""

                # Call Ollama API
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,  # Lower temp = more deterministic
                    max_tokens=100,
                )

                # Parse response
                try:
                    response_text = response.choices[0].message.content

                    # Extract JSON
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1

                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        action_dict = json.loads(json_str)

                        # Create action
                        action = OrderAction(
                            orders=action_dict.get("orders", {}),
                            emergency=action_dict.get("emergency", False),
                        )
                    else:
                        # Fallback: no order
                        action = OrderAction(orders={}, emergency=False)
                except Exception:
                    # Fallback on parse error
                    action = OrderAction(orders={}, emergency=False)

                # Execute action
                result = await env.step(action)
                episode_profit += result.reward

            # Get final state
            state = await env.state()

            scores.append(episode_profit)
            print(f"   ✓ Profit: {episode_profit:+.3f}, Revenue: ${state.total_revenue:.2f}, Cost: ${state.total_cost:.2f}")

        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
            scores.append(0.0)
        finally:
            # Cleanup: disconnect from server
            await env.disconnect()

    # Print results
    print("\n" + "="*70)
    print("   📊 FINAL RESULTS")
    print("="*70 + "\n")

    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"  EASY Difficulty")
        print(f"    Episodes: {len(scores)}")
        print(f"    Avg Profit: {avg_score:+.3f}")
        print(f"    Scores: {[f'{s:+.3f}' for s in scores]}")
    else:
        print(f"  No successful episodes")

    print("\n" + "="*70)
    print("✨ Baseline inference complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(baseline_agent())
