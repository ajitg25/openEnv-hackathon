#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Baseline inference script for Shop SKU Manager environment.

Uses OpenAI API to run an intelligent agent that learns inventory management.
Evaluates on all 3 difficulty levels.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/shop_sku_baseline.py
"""

import asyncio
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
    Baseline agent using OpenAI API.

    Agent reasons about:
    - Current inventory levels
    - Demand forecasts
    - Lead times
    - Budget constraints

    And makes order decisions to maximize profit.
    """
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)
    scores = {"easy": [], "medium": [], "hard": []}

    print("\n" + "="*70)
    print("🛒 SHOP SKU MANAGER - BASELINE INFERENCE")
    print("="*70 + "\n")

    for difficulty in ["easy", "medium", "hard"]:
        print(f"⚡ Testing {difficulty.upper()} difficulty...")
        print("   " + "─"*60)

        try:
            env = ShopSKUManagerEnv(base_url="http://localhost:8000")
        except Exception as e:
            print(f"\n❌ Error: Could not connect to server at http://localhost:8000")
            print(f"   Make sure the server is running:")
            print(f"   PYTHONPATH=src:envs SHOP_DIFFICULTY={difficulty} \\")
            print(f"     uv run uvicorn envs.shop_sku_manager.server.app:app --reload\n")
            sys.exit(1)

        for episode in range(5):  # 5 episodes per difficulty
            try:
                obs = await env.reset()
                episode_profit = 0.0
                day = 0

                while not obs.done:
                    day += 1

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

                    prompt = f"""
You are a shop inventory manager. Make ordering decisions to maximize profit.

Current Status (Day {day}):
- Inventory: {inventory_summary}
- Budget: ${obs.budget_remaining:.2f}
- Day of week: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][obs.day_of_week]}
- Season: {obs.season}

Forecasted demand for today:
{forecast_summary}

Lead times (days): {lead_times_summary}

Decision: What should we order today?
Respond ONLY with JSON format like:
{{"orders": {{"milk": 15, "bread": 10}}, "emergency": false}}

Or:
{{"orders": {{}}, "emergency": false}}

if you don't want to order anything.
"""

                    # Call OpenAI API
                    response = await client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=200,
                    )

                    # Parse response
                    try:
                        response_text = response.choices[0].message.content

                        # Extract JSON
                        import json
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
                    except Exception as e:
                        # Fallback on parse error
                        action = OrderAction(orders={}, emergency=False)

                    # Execute action
                    obs = await env.step(action)
                    episode_profit += obs.reward

                # Get final state
                state = await env.state()

                scores[difficulty].append(episode_profit)
                print(f"   Episode {episode+1}: profit={episode_profit:+.3f}, " +
                      f"revenue=${state.total_revenue:.2f}, cost=${state.total_cost:.2f}")

            except Exception as e:
                print(f"   Episode {episode+1}: ERROR - {str(e)}")
                scores[difficulty].append(0.0)

        print()

    # Print results
    print("="*70)
    print("   📊 FINAL RESULTS")
    print("="*70 + "\n")

    for difficulty in ["easy", "medium", "hard"]:
        avg_score = sum(scores[difficulty]) / len(scores[difficulty])
        print(f"  {difficulty.upper():10s}  Avg Score: {avg_score:+.3f}")

    print("\n" + "="*70)
    print("✨ Baseline inference complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(baseline_agent())
