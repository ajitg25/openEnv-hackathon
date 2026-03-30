#!/usr/bin/env python3
"""
Inference script for Shop SKU Manager environment.

This is the main entry point for hackathon evaluation.
Connects to the environment server and runs an LLM-powered agent
that makes inventory ordering decisions to maximize profit.

Usage:
    export OPENAI_API_KEY="sk-..."
    python inference.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add paths so imports work from repo root
sys.path.insert(0, str(Path(__file__).parent / "envs"))

from shop_sku_manager.client import ShopSKUManagerEnv
from shop_sku_manager.models import OrderAction

try:
    from openai import AsyncOpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)


SERVER_URL = os.getenv("OPENENV_SERVER_URL", "http://localhost:8000")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4")


async def run_episode(client: AsyncOpenAI, env: ShopSKUManagerEnv) -> float:
    """Run a single episode and return total profit."""
    result = await env.reset()
    episode_profit = 0.0
    day = 0

    while not result.done:
        day += 1
        obs = result.observation

        inventory_summary = ", ".join(
            f"{sku}: {int(qty)} units"
            for sku, qty in obs.inventory_levels.items()
        )
        forecast_summary = ", ".join(
            f"{sku}: {obs.demand_forecast[sku]:.1f}"
            for sku in obs.inventory_levels.keys()
        )
        lead_times_summary = ", ".join(
            f"{sku}: {obs.lead_times[sku]} days"
            for sku in obs.inventory_levels.keys()
        )

        prompt = f"""You are a shop inventory manager. Make ordering decisions to maximize profit.

Current Status (Day {day}):
- Inventory: {inventory_summary}
- Budget: ${obs.budget_remaining:.2f}
- Day of week: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][obs.day_of_week]}
- Season: {obs.season}
- Forecasted demand: {forecast_summary}
- Lead times: {lead_times_summary}

IMPORTANT: Respond ONLY with valid JSON. No other text.

Example valid responses:
{{"orders": {{"milk": 15, "bread": 10}}, "emergency": false}}
{{"orders": {{}}, "emergency": false}}

Your decision (JSON only):"""

        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
        )

        try:
            response_text = response.choices[0].message.content
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                action_dict = json.loads(response_text[json_start:json_end])
                action = OrderAction(
                    orders=action_dict.get("orders", {}),
                    emergency=action_dict.get("emergency", False),
                )
            else:
                action = OrderAction(orders={}, emergency=False)
        except Exception:
            action = OrderAction(orders={}, emergency=False)

        result = await env.step(action)
        episode_profit += result.reward

    return episode_profit


async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)

    print("\n" + "=" * 70)
    print("Shop SKU Manager - Inference")
    print("=" * 70)
    print(f"  Server: {SERVER_URL}")
    print(f"  Model:  {MODEL}")

    scores = {"easy": [], "medium": [], "hard": []}
    num_episodes = int(os.getenv("NUM_EPISODES", "5"))

    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n--- {difficulty.upper()} ({num_episodes} episodes) ---")

        for episode in range(num_episodes):
            env = ShopSKUManagerEnv(base_url=SERVER_URL)
            try:
                profit = await run_episode(client, env)
                scores[difficulty].append(profit)
                print(f"  Episode {episode + 1}: profit={profit:+.3f}")
            except Exception as e:
                print(f"  Episode {episode + 1}: ERROR - {e}")
                scores[difficulty].append(0.0)
            finally:
                await env.disconnect()

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for difficulty in ["easy", "medium", "hard"]:
        if scores[difficulty]:
            avg = sum(scores[difficulty]) / len(scores[difficulty])
            print(f"  {difficulty.upper():10s}  Avg: {avg:+.3f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
