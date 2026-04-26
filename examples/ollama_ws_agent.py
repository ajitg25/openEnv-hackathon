"""
Ollama WebSocket Agent for Ambulance Green Corridor
====================================================
Tests a local Ollama LLM via the OpenEnv WebSocket endpoint —
exactly how judges will evaluate the environment.

Usage:
    # Terminal 1 — start the environment server
    PYTHONPATH=envs python showcase.py

    # Terminal 2 — run the agent
    PYTHONPATH=envs python examples/ollama_ws_agent.py --model llama3.2 --episodes 3

The WebSocket endpoint is what OpenEnv's validator uses:
    ws://localhost:7860/ws
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "envs"))

from ambulance_env import AmbulanceEnv, AmbulanceAction, SignalControl

OLLAMA_URL = "http://localhost:11434"
ENV_URL    = "http://localhost:7860"   # showcase.py WebSocket server

SYSTEM_PROMPT = """You are an emergency services AI managing an ambulance in a city.

Your job each step:
1. If no hospital is chosen yet, pick the best one (consider specialization match + ETA + traffic volume).
2. Clear ONLY traffic signals marked WRONG. Never touch OK signals — unnecessary toggles cost reward.
3. If an alternative route has significantly lower ETA (heavy traffic blocks current route), switch hospitals.

Always reply with valid JSON and nothing else:
{"hospital_id": null, "signal_controls": [{"row": R, "col": C, "phase": "ns_green or ew_green"}], "preferred_direction": null}
"""


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(obs) -> str:
    lines = [
        "=== EMERGENCY DISPATCH ===",
        f"Patient   : {obs.patient_location} | condition: {obs.patient_condition}",
        f"Ambulance : {obs.ambulance_location} | time: {obs.time_elapsed_seconds:.0f}s / {obs.time_limit_seconds:.0f}s",
        "",
    ]
    if obs.active_events:
        lines.append("DYNAMIC EVENTS:")
        for e in obs.active_events:
            lines.append(f"  [{e.event_type.upper()}] at {e.position} — {e.description}")
        lines.append("")

    if obs.target_hospital_id and obs.current_route.estimated_time > 0:
        r = obs.current_route
        lines.append(f"CURRENT ROUTE → {obs.target_hospital_id}")
        lines.append(f"  ETA={r.estimated_time:.0f}s | damaged={r.num_damaged_segments} | heavy={r.num_heavy_traffic_segments}")
        lines.append("")

    if obs.alternative_routes:
        lines.append("ALTERNATIVES (switch if ETA much lower):")
        for alt in obs.alternative_routes:
            h = next((h for h in obs.hospitals if h.hospital_id == alt.hospital_id), None)
            spec = h.specialization if h else "?"
            match = " <- specialist" if h and h.specialization == obs.patient_condition else ""
            lines.append(f"  {alt.hospital_id} ({spec}){match}: ETA={alt.estimated_time:.0f}s | damaged={alt.num_damaged_segments}")
        lines.append("")

    lines.append("HOSPITALS:")
    for h in obs.hospitals:
        cap = " [AT CAPACITY]" if h.at_capacity else ""
        match = " <- specialist" if h.specialization == obs.patient_condition else ""
        lines.append(f"  {h.hospital_id}: {h.name} | spec={h.specialization} | est={h.travel_time_estimate:.0f}s{cap}{match}")
    lines.append("")

    if obs.lookahead_signals:
        lines.append("SIGNALS (only change WRONG):")
        for s in obs.lookahead_signals:
            needed = "ns_green" if s.ambulance_direction in ("north", "south") else "ew_green"
            status = "OK" if s.phase == needed else f"WRONG — needs {needed}"
            lines.append(f"  ({s.row},{s.col}): {s.phase} | dir={s.ambulance_direction} | {status}")
        lines.append("")

    if obs.current_segment:
        seg = obs.current_segment
        lines.append(f"ROAD: {seg.road_type} | traffic={seg.traffic_volume:.0%} | speed={obs.last_speed_factor:.0%}")
        lines.append("")

    lines.append(f"STATS: stops={obs.stops_at_red} | eff={obs.signal_efficiency:.0%} | wasted={obs.unnecessary_toggles}")
    lines.append("")
    lines.append('Reply JSON only: {"hospital_id": null, "signal_controls": [...], "preferred_direction": null}')
    return "\n".join(lines)


# ── Ollama call ───────────────────────────────────────────────────────────────
def call_ollama(model: str, prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=60,
    )
    return r.json()["message"]["content"]


# ── Action parser ─────────────────────────────────────────────────────────────
def parse_action(text: str, obs) -> AmbulanceAction:
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            hid = data.get("hospital_id")
            if hid:
                valid = {h.hospital_id for h in obs.hospitals if not h.at_capacity}
                if hid not in valid:
                    hid = None
            controls = [
                SignalControl(row=int(c["row"]), col=int(c["col"]), phase=c["phase"])
                for c in data.get("signal_controls", [])
                if isinstance(c, dict) and c.get("phase") in ("ns_green", "ew_green")
            ]
            d = data.get("preferred_direction")
            if d not in ("north", "south", "east", "west"):
                d = None
            return AmbulanceAction(hospital_id=hid, signal_controls=controls, preferred_direction=d)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass

    # Fallback
    if not obs.target_hospital_id:
        available = [h for h in obs.hospitals if not h.at_capacity]
        specs = [h for h in available if h.specialization == obs.patient_condition]
        pool = specs if specs else available
        if pool:
            return AmbulanceAction(hospital_id=min(pool, key=lambda h: h.travel_time_estimate).hospital_id)

    controls = [
        SignalControl(row=s.row, col=s.col,
                      phase="ns_green" if s.ambulance_direction in ("north", "south") else "ew_green")
        for s in obs.lookahead_signals
        if s.phase != ("ns_green" if s.ambulance_direction in ("north", "south") else "ew_green")
    ]
    return AmbulanceAction(signal_controls=controls)


# ── WebSocket episode ─────────────────────────────────────────────────────────
async def run_episode(model: str, verbose: bool = True) -> dict:
    async with AmbulanceEnv(base_url=ENV_URL) as env:
        result = await env.reset()
        obs = result.observation
        total_reward = 0.0
        step = 0

        if verbose:
            print(f"\n  Patient : {obs.patient_condition} at {obs.patient_location}")
            print(f"  Hospitals: {[h.hospital_id for h in obs.hospitals]}")

        while not result.done:
            prompt   = build_prompt(obs)
            response = call_ollama(model, prompt)
            action   = parse_action(response, obs)

            result = await env.step(action)
            obs = result.observation
            total_reward += result.reward or 0
            step += 1

            if verbose:
                hid   = action.hospital_id or "-"
                ctrls = len(action.signal_controls)
                evts  = " ".join(e.event_type for e in obs.active_events)
                print(f"  [{step:2d}] hosp={hid:10s} sigs={ctrls} "
                      f"r={result.reward:+7.1f} tot={total_reward:7.1f} "
                      f"eff={obs.signal_efficiency:.0%} {evts}")

        state = await env.state()
        return {
            "steps": step,
            "total_reward": total_reward,
            "success": state.success,
            "arrival_time": state.arrival_time,
            "signal_efficiency": state.signal_efficiency,
            "stops_at_red": state.total_stops,
            "unnecessary_toggles": obs.unnecessary_toggles,
            "successful_reroutes": state.successful_reroutes,
        }


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="llama3.2")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    # Preflight checks
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        print(f"Ollama   : {OLLAMA_URL} ✓")
    except Exception:
        print(f"ERROR: Ollama not running. Start with: ollama serve"); return

    try:
        requests.get(f"{ENV_URL}/api/health", timeout=3)
        print(f"Env      : {ENV_URL} (WebSocket) ✓")
    except Exception:
        print(f"ERROR: Env server not running. Start with: PYTHONPATH=envs python showcase.py"); return

    print(f"Model    : {args.model}")
    print(f"Episodes : {args.episodes}")
    print(f"WS URL   : ws://localhost:7860/ws\n")

    results = []
    for ep in range(1, args.episodes + 1):
        print(f"{'─'*55}")
        print(f"  Episode {ep}/{args.episodes}")
        r = await run_episode(args.model)
        results.append(r)
        print(f"\n  → {'ARRIVED' if r['success'] else 'TIMED OUT'} | "
              f"reward={r['total_reward']:.1f} | "
              f"eff={r['signal_efficiency']:.0%} | "
              f"stops={r['stops_at_red']}")

    print(f"\n{'='*55}")
    print(f"  SUMMARY  ({args.model}, {args.episodes} episodes)")
    print(f"{'='*55}")
    n = len(results)
    print(f"  Arrival rate    : {sum(r['success'] for r in results)}/{n}")
    print(f"  Mean reward     : {sum(r['total_reward'] for r in results)/n:.1f}")
    print(f"  Mean efficiency : {sum(r['signal_efficiency'] for r in results)/n:.0%}")
    print(f"  Mean stops      : {sum(r['stops_at_red'] for r in results)/n:.1f}")
    print(f"  Mean reroutes   : {sum(r['successful_reroutes'] for r in results)/n:.1f}")

if __name__ == "__main__":
    asyncio.run(main())
