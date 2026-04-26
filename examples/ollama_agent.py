"""
Ollama Agent for Ambulance Green Corridor
==========================================
Tests a local Ollama LLM against the environment.

Usage:
    # Terminal 1 — start the environment server
    PYTHONPATH=envs python showcase.py

    # Terminal 2 — run the agent
    python examples/ollama_agent.py --model llama3.2 --episodes 3

Requirements:
    pip install requests
    ollama pull llama3.2   (or any model you have)
"""

import argparse
import json
import re
import time

import requests

# ── Config ───────────────────────────────────────────────────────────────────
ENV_URL    = "http://localhost:7860"   # showcase.py
OLLAMA_URL = "http://localhost:11434"  # ollama serve

SYSTEM_PROMPT = """You are an emergency services AI managing an ambulance in a city.

Your job each step:
1. If no hospital is chosen yet, pick the best one (consider specialization match + ETA).
2. Clear traffic signals that are in the WRONG phase for the ambulance direction.
3. If an alternative route has a significantly lower ETA, switch hospitals.

RULES:
- Only change signals marked WRONG. Never touch OK signals.
- Avoid hospitals marked [AT CAPACITY] or [FULL].
- Prefer specialist hospitals that match the patient condition.
- If the current route ETA is much higher than an alternative, switch.

Always reply with valid JSON:
{"hospital_id": null, "signal_controls": [{"row": R, "col": C, "phase": "ns_green or ew_green"}], "preferred_direction": null}
"""


# ── Env API ──────────────────────────────────────────────────────────────────
def env_reset(difficulty="easy"):
    r = requests.post(f"{ENV_URL}/api/reset", json={"difficulty": difficulty}, timeout=10)
    return r.json()["observation"]


def env_step(action: dict):
    r = requests.post(f"{ENV_URL}/api/step", json=action, timeout=10)
    d = r.json()
    return d["observation"], d.get("reward", 0), d.get("done", False)


# ── Prompt builder ───────────────────────────────────────────────────────────
def build_prompt(obs: dict) -> str:
    lines = [
        "=== EMERGENCY DISPATCH ===",
        f"Patient   : {obs['patient_location']} | condition: {obs['patient_condition']}",
        f"Ambulance : {obs['ambulance_location']} | time: {obs['time_elapsed_seconds']:.0f}s / {obs['time_limit_seconds']:.0f}s",
        "",
    ]

    if obs.get("active_events"):
        lines.append("DYNAMIC EVENTS:")
        for e in obs["active_events"]:
            lines.append(f"  [{e['event_type'].upper()}] at {e['position']} — {e['description']}")
        lines.append("")

    if obs.get("target_hospital_id") and obs.get("current_route", {}).get("estimated_time"):
        r = obs["current_route"]
        lines.append(f"CURRENT ROUTE → {obs['target_hospital_id']}")
        lines.append(f"  ETA={r['estimated_time']:.0f}s | damaged={r['num_damaged_segments']} | heavy={r['num_heavy_traffic_segments']}")
        lines.append("")

    if obs.get("alternative_routes"):
        lines.append("ALTERNATIVES:")
        for alt in obs["alternative_routes"]:
            h = next((h for h in obs["hospitals"] if h["hospital_id"] == alt["hospital_id"]), None)
            spec = h["specialization"] if h else "?"
            match = " <- specialist match" if h and h["specialization"] == obs["patient_condition"] else ""
            lines.append(f"  {alt['hospital_id']} ({spec}){match}: ETA={alt['estimated_time']:.0f}s | damaged={alt['num_damaged_segments']}")
        lines.append("")

    lines.append("HOSPITALS:")
    for h in obs.get("hospitals", []):
        cap = " [AT CAPACITY]" if h.get("at_capacity") else ""
        match = " <- specialist match" if h["specialization"] == obs["patient_condition"] else ""
        lines.append(f"  {h['hospital_id']}: {h['name']} | spec={h['specialization']} | est={h['travel_time_estimate']:.0f}s{cap}{match}")
    lines.append("")

    if obs.get("lookahead_signals"):
        lines.append("SIGNALS (only change WRONG ones):")
        for s in obs["lookahead_signals"]:
            needed = "ns_green" if s["ambulance_direction"] in ("north", "south") else "ew_green"
            status = "OK" if s["phase"] == needed else f"WRONG — needs {needed}"
            lines.append(f"  ({s['row']},{s['col']}): current={s['phase']} | dir={s['ambulance_direction']} | {status}")
        lines.append("")

    lines.append(f"STATS: stops={obs.get('stops_at_red',0)} | efficiency={obs.get('signal_efficiency',1)*100:.0f}% | wasted={obs.get('unnecessary_toggles',0)}")
    lines.append("")
    lines.append('Reply with JSON: {"hospital_id": null, "signal_controls": [...], "preferred_direction": null}')

    return "\n".join(lines)


# ── Ollama call ───────────────────────────────────────────────────────────────
def call_ollama(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=60)
    return r.json()["message"]["content"]


# ── Action parser ─────────────────────────────────────────────────────────────
def parse_action(text: str, obs: dict) -> dict:
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            hid = data.get("hospital_id")
            valid_ids = {h["hospital_id"] for h in obs["hospitals"] if not h.get("at_capacity")}
            if hid and hid not in valid_ids:
                hid = None
            controls = [
                {"row": int(c["row"]), "col": int(c["col"]), "phase": c["phase"]}
                for c in data.get("signal_controls", [])
                if isinstance(c, dict) and c.get("phase") in ("ns_green", "ew_green")
            ]
            direction = data.get("preferred_direction")
            if direction not in ("north", "south", "east", "west"):
                direction = None
            return {"hospital_id": hid, "signal_controls": controls, "preferred_direction": direction}
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass

    # Fallback
    if not obs.get("target_hospital_id"):
        available = [h for h in obs["hospitals"] if not h.get("at_capacity")]
        specs = [h for h in available if h["specialization"] == obs["patient_condition"]]
        pool = specs if specs else available
        if pool:
            best = min(pool, key=lambda h: h["travel_time_estimate"])
            return {"hospital_id": best["hospital_id"], "signal_controls": []}

    controls = [
        {"row": s["row"], "col": s["col"],
         "phase": "ns_green" if s["ambulance_direction"] in ("north", "south") else "ew_green"}
        for s in obs.get("lookahead_signals", [])
        if s["phase"] != ("ns_green" if s["ambulance_direction"] in ("north", "south") else "ew_green")
    ]
    return {"hospital_id": None, "signal_controls": controls}


# ── Run one episode ───────────────────────────────────────────────────────────
def run_episode(model: str, difficulty: str, verbose: bool = True) -> dict:
    obs = env_reset(difficulty)
    total_reward = 0.0
    step = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Patient: {obs['patient_condition']} at {obs['patient_location']}")
        print(f"  Hospitals: {[h['hospital_id'] for h in obs['hospitals']]}")
        print(f"{'='*60}")

    while True:
        prompt   = build_prompt(obs)
        response = call_ollama(model, prompt)
        action   = parse_action(response, obs)

        obs, reward, done = env_step(action)
        total_reward += reward
        step += 1

        if verbose:
            hid   = action.get("hospital_id")
            ctrls = len(action.get("signal_controls", []))
            evts  = [e["event_type"] for e in obs.get("active_events", [])]
            print(f"  [{step:2d}] hosp={hid or '-':10s} sigs={ctrls} r={reward:+7.1f} "
                  f"tot={total_reward:7.1f} eff={obs.get('signal_efficiency',1)*100:.0f}% "
                  f"{' '.join(evts)}")

        if done:
            break

    result = {
        "steps": step,
        "total_reward": total_reward,
        "success": obs.get("done") and total_reward > 0,
        "arrival_time": obs.get("time_elapsed_seconds"),
        "signal_efficiency": obs.get("signal_efficiency", 0),
        "stops_at_red": obs.get("stops_at_red", 0),
        "unnecessary_toggles": obs.get("unnecessary_toggles", 0),
        "successful_reroutes": obs.get("successful_reroutes", 0),
    }

    if verbose:
        print(f"\n  Result  : {'ARRIVED' if result['success'] else 'TIMED OUT'}")
        print(f"  Reward  : {total_reward:.1f}")
        print(f"  Eff     : {result['signal_efficiency']*100:.0f}%")
        print(f"  Stops   : {result['stops_at_red']}")
        print(f"  Wasted  : {result['unnecessary_toggles']}")
        print(f"  Reroutes: {result['successful_reroutes']}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ollama agent for Ambulance Green Corridor")
    parser.add_argument("--model",      default="llama3.2",  help="Ollama model name")
    parser.add_argument("--episodes",   type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--difficulty", default="easy",       help="easy | medium | hard")
    args = parser.parse_args()

    # Check Ollama is running
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
    except Exception:
        print(f"ERROR: Ollama not running at {OLLAMA_URL}. Start with: ollama serve")
        return

    # Check env server
    try:
        requests.get(f"{ENV_URL}/api/health", timeout=3)
    except Exception:
        print(f"ERROR: Env server not running at {ENV_URL}. Start with: PYTHONPATH=envs python showcase.py")
        return

    print(f"\nModel     : {args.model}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Episodes  : {args.episodes}")

    results = []
    for ep in range(1, args.episodes + 1):
        print(f"\n--- Episode {ep}/{args.episodes} ---")
        r = run_episode(args.model, args.difficulty)
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY ({args.model}, {args.difficulty}, {args.episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Arrival rate   : {sum(r['success'] for r in results)}/{args.episodes}")
    print(f"  Mean reward    : {sum(r['total_reward'] for r in results)/len(results):.1f}")
    print(f"  Mean efficiency: {sum(r['signal_efficiency'] for r in results)/len(results)*100:.0f}%")
    print(f"  Mean stops     : {sum(r['stops_at_red'] for r in results)/len(results):.1f}")
    print(f"  Mean reroutes  : {sum(r['successful_reroutes'] for r in results)/len(results):.1f}")


if __name__ == "__main__":
    main()
