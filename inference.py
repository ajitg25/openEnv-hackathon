#!/usr/bin/env python3
"""
Inference script for Shop SKU Manager environment.

Connects to the environment server and runs an LLM-powered agent
that makes inventory ordering decisions to maximize profit.

Required env vars (injected by validator):
    API_BASE_URL   The LiteLLM proxy endpoint.
    API_KEY        The API key for the proxy.
    MODEL_NAME     The model identifier to use for inference.
"""

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent / "envs"))

from shop_sku_manager.client import ShopSKUManagerEnv
from shop_sku_manager.models import OrderAction

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SERVER_URL = os.getenv("OPENENV_SERVER_URL", "http://localhost:7860")

TASK_NAME = os.getenv("SHOP_SKU_TASK", "easy")
BENCHMARK = "shop_sku_manager"
MAX_STEPS = 30
TEMPERATURE = 0.3
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.1

MAX_TOTAL_REWARD = MAX_STEPS * 1.0


# ---------------------------------------------------------------------------
# Structured stdout logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM-powered ordering agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a shop inventory manager. Your goal is to maximize profit by making smart ordering decisions.\n"
    "\n"
    "Each turn you receive current inventory, demand forecasts, lead times, and budget.\n"
    "You must respond with ONLY valid JSON — no explanation, no markdown, no extra text.\n"
    "\n"
    'Format: {"orders": {"sku_name": quantity, ...}, "emergency": false}\n'
    'No orders: {"orders": {}, "emergency": false}\n'
    "\n"
    "Strategy:\n"
    "- Order when inventory < forecast * lead_time * 1.5 to avoid stockouts\n"
    "- Don't over-order — storage costs eat profit\n"
    "- Emergency shipping costs 50% more, use only if critically low\n"
    "- Respect budget constraints"
)


def build_prompt(obs) -> str:
    inventory = ", ".join(
        f"{sku}: {int(qty)} units" for sku, qty in obs.inventory_levels.items()
    )
    forecast = ", ".join(
        f"{sku}: {obs.demand_forecast[sku]:.1f}" for sku in obs.inventory_levels
    )
    lead_times = ", ".join(
        f"{sku}: {obs.lead_times[sku]}d" for sku in obs.inventory_levels
    )
    stockouts = [sku for sku, flag in obs.stockout_flags.items() if flag]
    stockout_str = ", ".join(stockouts) if stockouts else "none"

    return (
        f"Day {obs.current_day} ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][obs.day_of_week]}, {obs.season})\n"
        f"Inventory: {inventory}\n"
        f"Forecast: {forecast}\n"
        f"Lead times: {lead_times}\n"
        f"Budget: ${obs.budget_remaining:.2f}\n"
        f"Stockouts: {stockout_str}\n"
        f"Your order (JSON only):"
    )


def get_order(client: OpenAI, obs) -> OrderAction:
    prompt = build_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            action_dict = json.loads(text[json_start:json_end])
            return OrderAction(
                orders=action_dict.get("orders", {}),
                emergency=action_dict.get("emergency", False),
            )
    except Exception as e:
        print(f"[DEBUG] Model/parse error: {e}", flush=True)

    return OrderAction(orders={}, emergency=False)


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

async def main() -> None:
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[DEBUG] SERVER_URL={SERVER_URL}", flush=True)
    print(f"[DEBUG] API_KEY set={bool(API_KEY)}", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = ShopSKUManagerEnv(base_url=SERVER_URL)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        print("[DEBUG] Calling env.reset()...", flush=True)
        result = await env.reset()
        print(f"[DEBUG] env.reset() done. done={result.done}", flush=True)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                print(f"[DEBUG] Episode done at step {step}", flush=True)
                break

            obs = result.observation
            print(f"[DEBUG] Step {step}: calling LLM...", flush=True)
            action = get_order(client, obs)
            action_str = json.dumps(action.model_dump(), separators=(",", ":"))
            print(f"[DEBUG] Step {step}: LLM returned, calling env.step()...", flush=True)

            result = await env.step(action)

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        if rewards:
            score = sum(rewards) / MAX_TOTAL_REWARD
            score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        traceback.print_exc()

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
