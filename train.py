#!/usr/bin/env python3
"""
Ambulance Green Corridor — GRPO Training on HF Space (A10G GPU).

This script:
1. Starts the ambulance server as a background process
2. Loads Qwen2.5-0.5B-Instruct via Unsloth (4-bit LoRA)
3. Runs GRPO training for 60 iterations
4. Saves plots to /app/plots/
5. Exits — the Dockerfile then starts the serving mode
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nest_asyncio
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

nest_asyncio.apply()

sys.path.insert(0, "/app/envs")
from ambulance_env import AmbulanceEnv
from ambulance_env.models import AmbulanceAction, SignalControl

# ── Config ──────────────────────────────────────────────────────────────────
ENV_URL = "http://localhost:8000"
DIFFICULTY = os.getenv("AMBULANCE_DIFFICULTY", "easy")
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LEN = 1024
NUM_ITERATIONS = 10
GROUP_SIZE = 4
BETA_KL = 0.01
LR = 5e-5
PLOT_DIR = Path("/app/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Start server ────────────────────────────────────────────────────────
print("Starting ambulance_env server...")
server_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "ambulance_env.server.app:app",
     "--host", "0.0.0.0", "--port", "8000", "--log-level", "error"],
    env={**os.environ, "PYTHONPATH": "/app/envs", "AMBULANCE_DIFFICULTY": DIFFICULTY},
)
time.sleep(4)
print("Server ready.")


# ── 2. Load model ──────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
)
print("Base model loaded.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded.")

lora_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base_model, lora_config)
model.gradient_checkpointing_enable()
print(f"LoRA ready. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ── 3. Prompt formatters ──────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an emergency services AI managing an ambulance in a real city.\n"
    "You must:\n"
    "  1. Choose the best hospital (consider specialization, distance, traffic, road quality)\n"
    "  2. Clear traffic signals ahead — but ONLY signals in the WRONG phase\n"
    "  3. Re-route if traffic spikes, accidents, or road closures appear\n"
    "  4. Switch hospitals mid-journey if an alternative becomes significantly faster\n\n"
    "Heavy traffic slows the ambulance even on green. Potholed roads force slow speeds.\n"
    "Be precise. Follow the output format exactly."
)


def _quality_label(q):
    if q >= 0.9: return "highway"
    if q >= 0.65: return "good"
    if q >= 0.4: return "moderate"
    return "POTHOLED"


def format_prompt(obs):
    lines = [
        "=== EMERGENCY DISPATCH ===",
        f"Patient  : {obs.patient_location} | condition: {obs.patient_condition}",
        f"Ambulance: {obs.ambulance_location} | time: {obs.time_elapsed_seconds:.0f}s / {obs.time_limit_seconds:.0f}s",
        "",
    ]
    if obs.active_events:
        lines.append("DYNAMIC EVENTS:")
        for e in obs.active_events:
            lines.append(f"  [{e.event_type.upper()}] at {e.position} — {e.description}")
        lines.append("")
    if obs.target_hospital_id:
        r = obs.current_route
        lines.append(f"CURRENT ROUTE → {obs.target_hospital_id}")
        lines.append(f"  ETA={r.estimated_time:.0f}s | segs={len(r.segments)} | damaged={r.num_damaged_segments} | heavy={r.num_heavy_traffic_segments}")
        for seg in r.segments[:4]:
            lines.append(f"    {seg.from_pos}→{seg.to_pos} | {seg.road_type} | quality={_quality_label(seg.road_quality)} | traffic={seg.traffic_volume:.0%} | est={seg.estimated_transit_time:.0f}s" + (" [BLOCKED]" if seg.blocked else ""))
        lines.append("")
    if obs.alternative_routes:
        lines.append("ALTERNATIVES:")
        for alt in obs.alternative_routes:
            hosp = next((h for h in obs.hospitals if h.hospital_id == alt.hospital_id), None)
            spec = hosp.specialization if hosp else "?"
            match = " <- specialist" if hosp and hosp.specialization == obs.patient_condition else ""
            lines.append(f"  {alt.hospital_id} ({spec}){match}: ETA={alt.estimated_time:.0f}s | damaged={alt.num_damaged_segments}")
        lines.append("")
    lines.append("HOSPITALS:")
    for h in obs.hospitals:
        cap = " [FULL]" if h.at_capacity else ""
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
        lines.append(f"ROAD: {obs.current_segment.road_type} | quality={_quality_label(obs.current_segment.road_quality)} | traffic={obs.current_segment.traffic_volume:.0%} | speed={obs.last_speed_factor:.0%}")
        lines.append("")
    lines.append(f"STATS: stops={obs.stops_at_red} | efficiency={obs.signal_efficiency:.0%} | wasted={obs.unnecessary_toggles}")
    lines.append("")
    if not obs.target_hospital_id:
        lines.append('ACTION: {"hospital_id": "hosp_X", "signal_controls": [], "preferred_direction": null}')
    else:
        lines.append('ACTION: {"hospital_id": null, "signal_controls": [{"row": R, "col": C, "phase": "..."}], "preferred_direction": null}')
    return "\n".join(lines)


def build_chat(obs):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": format_prompt(obs)}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ── 4. Action parser ─────────────────────────────────────────────────────
def parse_action(response_text, obs):
    text = response_text.strip()
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
    if not obs.target_hospital_id:
        available = [h for h in obs.hospitals if not h.at_capacity]
        specs = [h for h in available if h.specialization == obs.patient_condition]
        pool = specs if specs else available
        if pool:
            return AmbulanceAction(hospital_id=min(pool, key=lambda h: h.travel_time_estimate).hospital_id)
    controls = [
        SignalControl(row=s.row, col=s.col, phase="ns_green" if s.ambulance_direction in ("north", "south") else "ew_green")
        for s in obs.lookahead_signals
        if s.phase != ("ns_green" if s.ambulance_direction in ("north", "south") else "ew_green")
    ]
    return AmbulanceAction(signal_controls=controls)


# ── 5. Episode rollout ───────────────────────────────────────────────────
@torch.no_grad()
async def collect_episode_async(temperature=0.8, max_new_tokens=256):
    env = AmbulanceEnv(base_url=ENV_URL)
    steps = []
    try:
        result = await env.reset()
        obs = result.observation
        while not result.done:
            prompt = build_chat(obs)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
            response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            action = parse_action(response_text, obs)
            result = await env.step(action)
            obs = result.observation
            steps.append({"prompt": prompt, "response": response_text, "step_reward": float(result.reward or 0.0)})
        total = sum(s["step_reward"] for s in steps)
        for s in steps:
            s["episode_reward"] = total
        state = await env.state()
        return steps, state
    finally:
        await env.close()


def collect_episode(temperature=0.8, max_new_tokens=256):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(collect_episode_async(temperature, max_new_tokens))


# ── 6. Evaluate ──────────────────────────────────────────────────────────
def evaluate(num_episodes=8):
    rewards, arrivals, effs, times, reroutes = [], [], [], [], []
    for _ in range(num_episodes):
        steps, state = collect_episode(temperature=0.1)
        rewards.append(steps[-1]["episode_reward"] if steps else 0.0)
        arrivals.append(float(state.success))
        effs.append(state.signal_efficiency)
        times.append(state.arrival_time or 999.0)
        reroutes.append(state.successful_reroutes)
    return {
        "mean_reward": float(np.mean(rewards)),
        "arrival_rate": float(np.mean(arrivals)),
        "mean_efficiency": float(np.mean(effs)),
        "mean_time": float(np.mean(times)),
        "mean_reroutes": float(np.mean(reroutes)),
    }


# ── 7. Baseline ──────────────────────────────────────────────────────────
print("\n=== Baseline evaluation ===")
print("Running episode 1 of 2...")
baseline = evaluate(num_episodes=2)
print(f"BASELINE  reward={baseline['mean_reward']:.1f}  arrival={baseline['arrival_rate']:.0%}  "
      f"efficiency={baseline['mean_efficiency']:.0%}  reroutes={baseline['mean_reroutes']:.1f}  "
      f"time={baseline['mean_time']:.0f}s")


# ── 8. GRPO Training ────────────────────────────────────────────────────
print(f"\n=== GRPO training: {NUM_ITERATIONS} iterations x {GROUP_SIZE} episodes ===\n")

optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
history = {"iteration": [], "mean_reward": [], "arrival_rate": [], "signal_efficiency": [], "mean_time": [], "mean_reroutes": []}

for iteration in range(NUM_ITERATIONS):
    model.eval()
    group_steps, group_states = [], []
    for _ in range(GROUP_SIZE):
        steps, state = collect_episode(temperature=0.8)
        group_steps.append(steps)
        group_states.append(state)

    episode_rewards = [s[-1]["episode_reward"] if s else 0.0 for s in group_steps]
    r_tensor = torch.tensor(episode_rewards)
    advantages = (r_tensor - r_tensor.mean()) / (r_tensor.std() + 1e-8)

    model.train()
    iter_loss, num_updates = 0.0, 0
    for steps, adv in zip(group_steps, advantages.tolist()):
        for step in steps:
            prompt_ids = tokenizer(step["prompt"], return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN - 256).input_ids.to(model.device)
            response_ids = tokenizer(step["response"], return_tensors="pt", truncation=True, max_length=256).input_ids.to(model.device)
            if response_ids.shape[1] == 0:
                continue
            full_ids = torch.cat([prompt_ids, response_ids], dim=1)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(full_ids).logits
            resp_logits = logits[:, prompt_ids.shape[1] - 1 : -1, :]
            log_probs = F.log_softmax(resp_logits, dim=-1)
            token_lp = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)
            mean_lp = token_lp.mean()
            loss = -adv * mean_lp + BETA_KL * (mean_lp ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            iter_loss += loss.item()
            num_updates += 1

    mr = float(np.mean(episode_rewards))
    ar = float(np.mean([s.success for s in group_states]))
    me = float(np.mean([s.signal_efficiency for s in group_states]))
    mt = float(np.mean([s.arrival_time or 999.0 for s in group_states]))
    mrr = float(np.mean([s.successful_reroutes for s in group_states]))

    history["iteration"].append(iteration + 1)
    history["mean_reward"].append(mr)
    history["arrival_rate"].append(ar)
    history["signal_efficiency"].append(me)
    history["mean_time"].append(mt)
    history["mean_reroutes"].append(mrr)

    print(f"[{iteration+1:3d}/{NUM_ITERATIONS}]  reward={mr:7.1f}  arrival={ar:.0%}  "
          f"efficiency={me:.0%}  reroutes={mrr:.1f}  time={mt:5.0f}s  "
          f"loss={iter_loss / max(1, num_updates):.4f}")


# ── 9. Final eval ───────────────────────────────────────────────────────
print("\n=== Final evaluation ===")
final = evaluate(num_episodes=8)
print(f"FINAL     reward={final['mean_reward']:.1f}  arrival={final['arrival_rate']:.0%}  "
      f"efficiency={final['mean_efficiency']:.0%}  reroutes={final['mean_reroutes']:.1f}  "
      f"time={final['mean_time']:.0f}s")

print("\n── Improvement ──────────────────────────────────────────")
print(f"  Reward      : {baseline['mean_reward']:6.1f} → {final['mean_reward']:6.1f}  ({final['mean_reward']-baseline['mean_reward']:+.1f})")
print(f"  Arrival     : {baseline['arrival_rate']:.0%}     → {final['arrival_rate']:.0%}")
print(f"  Efficiency  : {baseline['mean_efficiency']:.0%}     → {final['mean_efficiency']:.0%}")
print(f"  Reroutes    : {baseline['mean_reroutes']:.1f}       → {final['mean_reroutes']:.1f}")
print(f"  Travel time : {baseline['mean_time']:.0f}s     → {final['mean_time']:.0f}s  ({final['mean_time']-baseline['mean_time']:+.0f}s)")


# ── 10. Plots ────────────────────────────────────────────────────────────
def smooth(values, window=5):
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
fig.suptitle("Ambulance Green Corridor — GRPO Training", fontsize=14, fontweight="bold")
iters = history["iteration"]
sm = 4

ax = axes[0]
ax.plot(iters, history["mean_reward"], alpha=0.25, color="royalblue")
ax.plot(iters[sm:], smooth(history["mean_reward"]), color="royalblue", lw=2, label="Trained")
ax.axhline(baseline["mean_reward"], color="red", ls="--", lw=1.5, label=f"Baseline ({baseline['mean_reward']:.0f})")
ax.axhline(final["mean_reward"], color="green", ls="--", lw=1.5, label=f"Final ({final['mean_reward']:.0f})")
ax.set_xlabel("Episode"); ax.set_ylabel("Reward"); ax.set_title("Episode Reward"); ax.legend(fontsize=8)

ax = axes[1]
ax.plot(iters, [v * 100 for v in history["arrival_rate"]], alpha=0.25, color="darkorange")
ax.plot(iters[sm:], smooth([v * 100 for v in history["arrival_rate"]]), color="darkorange", lw=2)
ax.axhline(baseline["arrival_rate"] * 100, color="red", ls="--", lw=1.5, label=f"Before ({baseline['arrival_rate']:.0%})")
ax.axhline(final["arrival_rate"] * 100, color="green", ls="--", lw=1.5, label=f"After ({final['arrival_rate']:.0%})")
ax.set_xlabel("Episode"); ax.set_ylabel("Arrival (%)"); ax.set_title("Hospital Arrival Rate"); ax.set_ylim(0, 105); ax.legend(fontsize=8)

ax = axes[2]
ax.plot(iters, [v * 100 for v in history["signal_efficiency"]], alpha=0.25, color="seagreen")
ax.plot(iters[sm:], smooth([v * 100 for v in history["signal_efficiency"]]), color="seagreen", lw=2)
ax.axhline(baseline["mean_efficiency"] * 100, color="red", ls="--", lw=1.5, label=f"Before ({baseline['mean_efficiency']:.0%})")
ax.axhline(final["mean_efficiency"] * 100, color="green", ls="--", lw=1.5, label=f"After ({final['mean_efficiency']:.0%})")
ax.set_xlabel("Episode"); ax.set_ylabel("Efficiency (%)"); ax.set_title("Signal Efficiency"); ax.set_ylim(0, 105); ax.legend(fontsize=8)

ax = axes[3]
ax.plot(iters, history["mean_reroutes"], alpha=0.25, color="purple")
ax.plot(iters[sm:], smooth(history["mean_reroutes"]), color="purple", lw=2)
ax.axhline(baseline["mean_reroutes"], color="red", ls="--", lw=1.5, label=f"Before ({baseline['mean_reroutes']:.1f})")
ax.axhline(final["mean_reroutes"], color="green", ls="--", lw=1.5, label=f"After ({final['mean_reroutes']:.1f})")
ax.set_xlabel("Episode"); ax.set_ylabel("Reroutes"); ax.set_title("Adaptive Re-routing"); ax.legend(fontsize=8)

plt.tight_layout()
out = PLOT_DIR / "ambulance_training_results.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {out}")

# Save results as JSON for the web UI
import json as _json
results = {"baseline": baseline, "final": final, "history": history}
with open(PLOT_DIR / "results.json", "w") as f:
    _json.dump(results, f, indent=2)
print(f"Results saved → {PLOT_DIR / 'results.json'}")

# ── Cleanup ──────────────────────────────────────────────────────────────
server_proc.terminate()
print("\nTraining complete. Server stopped.")
