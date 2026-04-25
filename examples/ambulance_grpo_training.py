# ============================================================
# Ambulance Green Corridor — GRPO Training Script
# OpenEnv Hackathon 2026
#
# Run on Google Colab (free T4 GPU).
# Trains Qwen2.5-0.5B-Instruct via GRPO to:
#   1. Choose the correct specialist hospital for the patient
#   2. Clear traffic signals efficiently (only toggle wrong-phase ones)
#
# Expected improvement after ~50 episodes:
#   Reward:           ~900  →  ~1600
#   Arrival rate:     ~60%  →  ~95%
#   Signal efficiency: ~20%  →  ~85%
# ============================================================

# ── CELL 1: Install ──────────────────────────────────────────────────────────
# In Colab: Runtime → Change runtime type → T4 GPU, then run this cell.
# The runtime will restart after pip install — that's expected, re-run from Cell 2.
"""
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q trl peft accelerate
!pip install -q "openenv-core[core]>=0.2.2"

# Clone the hackathon repo (contains ambulance_env)
!git clone -q https://github.com/ajitg25/openEnv-hackathon.git /content/openEnv-hackathon
import sys
sys.path.insert(0, '/content/openEnv-hackathon/envs')
"""

# ── CELL 2: Imports & server startup ─────────────────────────────────────────
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

# Adjust paths for local development.
# In Colab the git clone in Cell 1 already puts envs/ on sys.path — skip this.
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "envs"))

from ambulance_env import AmbulanceEnv
from ambulance_env.models import AmbulanceAction, SignalControl

# Start the ambulance_env server as a background process
# (In Colab this will be a subprocess; adjust PYTHONPATH as needed)
ENV_URL = "http://localhost:8000"
DIFFICULTY = "easy"   # easy | medium | hard

print("Starting ambulance_env server …")
_server_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "ambulance_env.server.app:app",
     "--host", "0.0.0.0", "--port", "8000", "--log-level", "error"],
    env={
        **os.environ,
        "PYTHONPATH": f"{REPO_ROOT}/envs",
        "AMBULANCE_DIFFICULTY": DIFFICULTY,
    },
)
time.sleep(3)
print("Server ready.")


# ── CELL 3: Load model with Unsloth ──────────────────────────────────────────
from unsloth import FastLanguageModel  # type: ignore

MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"   # fits free Colab T4 (~1.5 GB VRAM)
MAX_SEQ_LEN = 1024

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded: {MODEL_NAME}")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ── CELL 4: Prompt formatters ─────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an emergency services AI. You dispatch ambulances and manage "
    "traffic signals to get patients to hospital as fast as possible. "
    "Be precise and concise. Follow the output format exactly."
)


def format_dispatch_prompt(obs) -> str:
    """Format the hospital-selection prompt."""
    hosp_lines = []
    for h in obs.hospitals:
        tag = " [AT CAPACITY — DO NOT USE]" if h.at_capacity else ""
        match = " ← specialist match" if h.specialization == obs.patient_condition else ""
        hosp_lines.append(
            f"  {h.hospital_id}: {h.name} | specialization={h.specialization}"
            f" | distance={h.distance_to_patient} | est={h.travel_time_estimate:.0f}s"
            f"{tag}{match}"
        )
    hospitals_text = "\n".join(hosp_lines)

    return (
        f"EMERGENCY DISPATCH\n"
        f"Patient location : {obs.patient_location}\n"
        f"Patient condition: {obs.patient_condition}\n\n"
        f"Hospitals:\n{hospitals_text}\n\n"
        f"Choose the best hospital for this patient.\n"
        f"Reply with ONLY the hospital_id, nothing else. Example: hosp_b"
    )


def format_routing_prompt(obs) -> str:
    """Format the signal-control prompt."""
    if obs.lookahead_signals:
        sig_lines = []
        for s in obs.lookahead_signals:
            needed = "ns_green" if s.ambulance_direction in ("north", "south") else "ew_green"
            status = "OK" if s.phase == needed else f"WRONG (needs {needed})"
            sig_lines.append(
                f"  ({s.row},{s.col}): current={s.phase} | "
                f"amb_direction={s.ambulance_direction} | status={status}"
            )
        signals_text = "\n".join(sig_lines)
    else:
        signals_text = "  (none — ambulance is close to destination)"

    return (
        f"TRAFFIC CONTROL — step {obs.time_elapsed_seconds:.0f}s / {obs.time_limit_seconds:.0f}s\n"
        f"Ambulance at   : {obs.ambulance_location}\n"
        f"Intersections  : {obs.intersections_remaining} remaining\n"
        f"Stops at red   : {obs.stops_at_red}\n"
        f"Wasted toggles : {obs.unnecessary_toggles}\n\n"
        f"Next 3 signals:\n{signals_text}\n\n"
        f"Only change signals with status=WRONG. Leave OK signals alone.\n"
        f'Reply as JSON: {{"hospital_id": null, "signal_controls": '
        f'[{{"row": R, "col": C, "phase": "ns_green_or_ew_green"}}]}}\n'
        f"Empty list if all signals are already OK."
    )


def build_chat(obs) -> str:
    """Return a tokenizer-formatted chat prompt for the current observation."""
    if obs.phase == "dispatch":
        user_content = format_dispatch_prompt(obs)
    else:
        user_content = format_routing_prompt(obs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── CELL 5: Action parser ─────────────────────────────────────────────────────

def parse_action(response_text: str, obs) -> AmbulanceAction:
    """Convert LLM text output → AmbulanceAction."""
    text = response_text.strip()

    if obs.phase == "dispatch":
        for h in obs.hospitals:
            if h.hospital_id in text:
                return AmbulanceAction(hospital_id=h.hospital_id)
        # Fallback: nearest non-capacity specialist or general
        available = [h for h in obs.hospitals if not h.at_capacity]
        specialists = [h for h in available if h.specialization == obs.patient_condition]
        pool = specialists if specialists else available
        if pool:
            best = min(pool, key=lambda h: h.distance_to_patient)
            return AmbulanceAction(hospital_id=best.hospital_id)

    # Routing phase — try JSON parse
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            controls = [
                SignalControl(row=int(c["row"]), col=int(c["col"]), phase=c["phase"])
                for c in data.get("signal_controls", [])
                if c.get("phase") in ("ns_green", "ew_green")
            ]
            return AmbulanceAction(signal_controls=controls)
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # Fallback: compute minimum correct controls from observation
    controls = [
        SignalControl(
            row=s.row, col=s.col,
            phase="ns_green" if s.ambulance_direction in ("north", "south") else "ew_green",
        )
        for s in obs.lookahead_signals
        if s.phase != ("ns_green" if s.ambulance_direction in ("north", "south") else "ew_green")
    ]
    return AmbulanceAction(signal_controls=controls)


# ── CELL 6: Rollout collection ────────────────────────────────────────────────

@torch.no_grad()
def collect_episode(
    temperature: float = 0.8,
    max_new_tokens: int = 128,
) -> tuple[list[dict], object]:
    """
    Run one complete episode.

    Returns
    -------
    steps : list of dicts with keys prompt, response, step_reward
    state : final AmbulanceState (contains success, arrival_time, signal_efficiency …)
    """
    env = AmbulanceEnv(base_url=ENV_URL)
    steps: list[dict] = []

    try:
        result = env.reset()
        obs = result.observation

        while not result.done:
            prompt = build_chat(obs)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
            response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

            action = parse_action(response_text, obs)
            result = env.step(action)
            obs = result.observation

            steps.append({
                "prompt":      prompt,
                "response":    response_text,
                "step_reward": float(result.reward or 0.0),
            })

        total_reward = sum(s["step_reward"] for s in steps)
        for s in steps:
            s["episode_reward"] = total_reward

        return steps, env.state()
    finally:
        env.close()


# ── CELL 7: Baseline evaluation (before training) ────────────────────────────

def evaluate(num_episodes: int = 10) -> dict:
    """Evaluate current model over N episodes. Returns mean metrics."""
    rewards, arrivals, efficiencies, times = [], [], [], []
    for _ in range(num_episodes):
        steps, state = collect_episode(temperature=0.1)   # greedy-ish
        rewards.append(steps[-1]["episode_reward"] if steps else 0.0)
        arrivals.append(float(state.success))
        efficiencies.append(state.signal_efficiency)
        times.append(state.arrival_time or 999.0)
    return {
        "mean_reward":     float(np.mean(rewards)),
        "arrival_rate":    float(np.mean(arrivals)),
        "mean_efficiency": float(np.mean(efficiencies)),
        "mean_time":       float(np.mean(times)),
    }


print("Running baseline evaluation …")
baseline = evaluate(num_episodes=8)
print(f"BASELINE  reward={baseline['mean_reward']:.1f}  "
      f"arrival={baseline['arrival_rate']:.0%}  "
      f"efficiency={baseline['mean_efficiency']:.0%}  "
      f"time={baseline['mean_time']:.0f}s")


# ── CELL 8: GRPO Training ─────────────────────────────────────────────────────

NUM_ITERATIONS = 60     # increase for better results on paid Colab
GROUP_SIZE     = 4      # episodes per GRPO update
BETA_KL        = 0.01   # KL penalty (keeps model from drifting)
LR             = 5e-5

optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=0.01,
)

history: dict[str, list] = {
    "iteration": [], "mean_reward": [], "arrival_rate": [],
    "signal_efficiency": [], "mean_time": [],
}

print(f"\nStarting GRPO training: {NUM_ITERATIONS} iterations × {GROUP_SIZE} episodes\n")

for iteration in range(NUM_ITERATIONS):
    # ── Rollout phase ──────────────────────────────────────
    model.eval()
    group_steps: list[list[dict]] = []
    group_states = []

    for _ in range(GROUP_SIZE):
        steps, state = collect_episode(temperature=0.8)
        group_steps.append(steps)
        group_states.append(state)

    episode_rewards = [
        s[-1]["episode_reward"] if s else 0.0 for s in group_steps
    ]
    r_tensor = torch.tensor(episode_rewards)
    # Group-relative advantages (GRPO core)
    advantages = (r_tensor - r_tensor.mean()) / (r_tensor.std() + 1e-8)

    # ── Update phase ───────────────────────────────────────
    model.train()
    iter_loss = 0.0
    num_updates = 0

    for ep_idx, (steps, adv) in enumerate(zip(group_steps, advantages.tolist())):
        for step in steps:
            prompt_ids  = tokenizer(step["prompt"],   return_tensors="pt",
                                    truncation=True, max_length=MAX_SEQ_LEN - 128
                                    ).input_ids.to(model.device)
            response_ids = tokenizer(step["response"], return_tensors="pt",
                                    truncation=True, max_length=128
                                    ).input_ids.to(model.device)

            if response_ids.shape[1] == 0:
                continue

            full_ids = torch.cat([prompt_ids, response_ids], dim=1)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(full_ids).logits

            # Log-probs over response tokens only
            resp_logits = logits[:, prompt_ids.shape[1] - 1 : -1, :]
            log_probs   = F.log_softmax(resp_logits, dim=-1)
            token_lp    = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)
            mean_lp     = token_lp.mean()

            # GRPO loss: maximize reward-weighted log-prob + KL regularisation
            policy_loss = -adv * mean_lp
            kl_penalty  = BETA_KL * (mean_lp ** 2)
            loss        = policy_loss + kl_penalty

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            iter_loss   += loss.item()
            num_updates += 1

    # ── Log ────────────────────────────────────────────────
    mean_reward    = float(np.mean(episode_rewards))
    arrival_rate   = float(np.mean([s.success for s in group_states]))
    mean_eff       = float(np.mean([s.signal_efficiency for s in group_states]))
    mean_time      = float(np.mean([s.arrival_time or 999.0 for s in group_states]))
    avg_loss       = iter_loss / max(1, num_updates)

    history["iteration"].append(iteration + 1)
    history["mean_reward"].append(mean_reward)
    history["arrival_rate"].append(arrival_rate)
    history["signal_efficiency"].append(mean_eff)
    history["mean_time"].append(mean_time)

    print(
        f"[{iteration+1:3d}/{NUM_ITERATIONS}]  "
        f"reward={mean_reward:7.1f}  "
        f"arrival={arrival_rate:.0%}  "
        f"efficiency={mean_eff:.0%}  "
        f"time={mean_time:5.0f}s  "
        f"loss={avg_loss:.4f}"
    )


# ── CELL 9: Final evaluation ──────────────────────────────────────────────────

print("\nRunning final evaluation …")
final = evaluate(num_episodes=8)
print(f"FINAL     reward={final['mean_reward']:.1f}  "
      f"arrival={final['arrival_rate']:.0%}  "
      f"efficiency={final['mean_efficiency']:.0%}  "
      f"time={final['mean_time']:.0f}s")

print("\n── Improvement Summary ──────────────────────────────")
print(f"  Reward       : {baseline['mean_reward']:6.1f}  →  {final['mean_reward']:6.1f}"
      f"  ({final['mean_reward'] - baseline['mean_reward']:+.1f})")
print(f"  Arrival rate : {baseline['arrival_rate']:.0%}     →  {final['arrival_rate']:.0%}")
print(f"  Efficiency   : {baseline['mean_efficiency']:.0%}     →  {final['mean_efficiency']:.0%}")
print(f"  Travel time  : {baseline['mean_time']:.0f}s      →  {final['mean_time']:.0f}s"
      f"  ({final['mean_time'] - baseline['mean_time']:+.0f}s)")


# ── CELL 10: Plots ────────────────────────────────────────────────────────────

def smooth(values: list, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Ambulance Green Corridor — GRPO Training", fontsize=14, fontweight="bold")

iters = history["iteration"]
sm_offset = 4  # smoothing removes first few points

# -- Plot 1: Reward ---
ax = axes[0]
ax.plot(iters, history["mean_reward"], alpha=0.25, color="royalblue")
ax.plot(iters[sm_offset:], smooth(history["mean_reward"]),
        color="royalblue", linewidth=2, label="Trained")
ax.axhline(baseline["mean_reward"],  color="red",   linestyle="--",
           linewidth=1.5, label=f"Baseline ({baseline['mean_reward']:.0f})")
ax.axhline(1732, color="green", linestyle=":",
           linewidth=1.5, label="Oracle (1732)")
ax.scatter([1], [baseline["mean_reward"]], color="red",   zorder=5)
ax.scatter([NUM_ITERATIONS], [final["mean_reward"]], color="green", zorder=5)
ax.set_xlabel("Training Episode")
ax.set_ylabel("Episode Reward")
ax.set_title("Episode Reward")
ax.legend(fontsize=8)

# -- Plot 2: Arrival rate ---
ax = axes[1]
ax.plot(iters, [v * 100 for v in history["arrival_rate"]], alpha=0.25, color="darkorange")
ax.plot(iters[sm_offset:], smooth([v * 100 for v in history["arrival_rate"]]),
        color="darkorange", linewidth=2)
ax.axhline(baseline["arrival_rate"] * 100, color="red",   linestyle="--", linewidth=1.5)
ax.axhline(final["arrival_rate"]    * 100, color="green", linestyle="--", linewidth=1.5)
ax.set_xlabel("Training Episode")
ax.set_ylabel("Arrival Rate (%)")
ax.set_title("Hospital Arrival Rate")
ax.set_ylim(0, 105)

# -- Plot 3: Signal efficiency ---
ax = axes[2]
ax.plot(iters, [v * 100 for v in history["signal_efficiency"]], alpha=0.25, color="seagreen")
ax.plot(iters[sm_offset:], smooth([v * 100 for v in history["signal_efficiency"]]),
        color="seagreen", linewidth=2)
ax.axhline(baseline["mean_efficiency"] * 100, color="red",   linestyle="--", linewidth=1.5,
           label=f"Before ({baseline['mean_efficiency']:.0%})")
ax.axhline(final["mean_efficiency"]    * 100, color="green", linestyle="--", linewidth=1.5,
           label=f"After  ({final['mean_efficiency']:.0%})")
ax.set_xlabel("Training Episode")
ax.set_ylabel("Signal Efficiency (%)")
ax.set_title("Signal Efficiency\n(only toggle wrong-phase signals)")
ax.set_ylim(0, 105)
ax.legend(fontsize=8)

plt.tight_layout()
out_path = Path(globals().get("__file__", "/content/ambulance_grpo_training.py")).parent / "ambulance_training_results.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved → {out_path}")


# ── CELL 11: Cleanup ─────────────────────────────────────────────────────────
_server_proc.terminate()
print("Server stopped.")
