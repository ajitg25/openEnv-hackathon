# Ambulance Green Corridor — OpenEnv Hackathon 2026

## Theme: #3.1 — World Modeling / Professional Tasks

**One-line summary:** We train an LLM to act as an emergency dispatcher + city traffic signal manager, navigating a partially observable city with real-world constraints (gridlock, potholes, accidents, hospital capacity) to get ambulances to the right hospital as fast as possible.

**Demo video:** [Watch on YouTube](https://youtu.be/9O5z4IXXtcc)

---

## The Capability Gap We Are Targeting

Current GPS-based emergency preemption systems (like Opticom, used globally) work like this:

> Ambulance within 300m of an intersection → turn it green.

That is reactive. It has no awareness of:
- What road quality lies ahead (potholed roads slow the ambulance even on green)
- Whether the nearest hospital is the *right* hospital for this patient's condition
- Whether heavy traffic on the planned route makes a longer detour actually faster
- Dynamic events mid-journey: accidents, road closures, traffic spikes

**The question we ask:** Can an LLM reason about the full journey — hospital selection, road quality, live traffic state, and mid-episode events — to get ambulances to the right place faster than any rule?

This is a genuine professional task that requires **persistent world modeling** across many steps. A rule cannot solve it. A shortest-path algorithm cannot solve it. This environment tests whether an LLM can.

---

## Theme Alignment: Why This Is Theme 3.1

Theme 3.1 asks for environments where:
> "the model is expected to do real hard work instead of exploiting short-cuts"

Our environment prevents shortcuts in three ways:

1. **Toggling already-green signals costs reward.** The agent must *read* signal state before acting — it cannot blindly clear everything.
2. **Traffic volume slows the ambulance even on green.** The agent cannot just clear signals and assume it will go fast — it must reason about the traffic volume on each segment.
3. **The nearest hospital is not always correct.** A cardiac patient sent to a trauma centre loses the +300 specialist bonus. But even the right specialist hospital may not be the best choice — if the route to it is gridlocked, clearing signals only gets you 20% speed through dense traffic. A farther hospital with lighter traffic and a lower ETA is the smarter pick. The agent must weigh specialization + distance + live traffic volume simultaneously, and be willing to switch hospitals mid-journey if conditions change.

The agent must maintain a coherent world model across 15–30 steps, update it when dynamic events fire, and make non-obvious decisions that only pay off several steps later.

---

## Environment Design

### What the Agent Sees (Observation)

Every step, the agent receives a structured observation:

```
=== EMERGENCY DISPATCH ===
Patient  : (6, 3) | condition: cardiac
Ambulance: (6, 4) | time: 40s / 300s

⚠ DYNAMIC EVENTS:
  [ACCIDENT] at (4,3) — road blocked (severity=0.8)

CURRENT ROUTE → hosp_a (City General)
  ETA=251s | segments=8 | damaged=2 | heavy_traffic=1
  (6,4)→(5,4) | residential | quality=moderate | traffic=45%
  (5,4)→(4,4) | damaged     | quality=POTHOLED | [BLOCKED]

ALTERNATIVES:
  hosp_c (Cardiac Centre) ← specialist match | ETA=130s | damaged=0

SIGNALS — only change WRONG ones:
  (5,4): ns_green | ambulance going north | OK
  (4,4): ew_green | ambulance going north | WRONG — needs ns_green

ACTION FORMAT:
{"hospital_id": "hosp_c", "signal_controls": [{"row": 4, "col": 4, "phase": "ns_green"}], "preferred_direction": null}
```

### What the Agent Does (Action Space)

```json
{
  "hospital_id": "hosp_c",
  "signal_controls": [{"row": 4, "col": 4, "phase": "ns_green"}],
  "preferred_direction": "north"
}
```

- `hospital_id`: choose or switch destination at any step (not just at the start)
- `signal_controls`: override up to 3 signals in the lookahead window
- `preferred_direction`: hint the routing engine to take a specific turn

### Reward Function — Designed to Be Hard to Game

| Component | Value | Purpose |
|---|---|---|
| Arrival | +1000 | Primary objective |
| Time bonus | +500 max | Rewards speed |
| Specialist match | +300 | Rewards reading patient condition |
| Red light stop | −20 each | Penalises poor signal management |
| **Unnecessary toggle** | **−2/−5 each** | **Core anti-shortcut mechanism** |
| Damaged road traversed | −10 each | Rewards road quality awareness |
| Successful re-route | +50 each | Rewards dynamic adaptation |

**The unnecessary toggle penalty is the key design decision.** An agent that blindly clears every signal in view scores *lower* than one that reads the state first. This forces genuine reasoning, not pattern-matching.

### Difficulty Levels

| Level | Grid | Hospitals | Base Traffic | Events/Step | Time Limit |
|---|---|---|---|---|---|
| easy | 6×6 | 2 general | Low (0.1) | 5% | 200s |
| medium | 8×8 | 3 mixed | Moderate (0.3) | 10% | 300s |
| hard | 12×12 | 5 (1 at capacity) | Heavy (0.5) | 15% | 400s |

---

## Training

**Model:** `Qwen/Qwen2.5-0.5B-Instruct` + LoRA (r=16, 2.1M trainable params)  
**Algorithm:** GRPO (Group Relative Policy Optimisation) via HuggingFace TRL  
**Setup:** 10 iterations × 4 episodes per iteration, live environment connection

### Results

![Training curves — reward, arrival rate, signal efficiency, re-routing](ambulance_training_results.png)

| Metric | Baseline (untrained) | After Training | Change |
|---|---|---|---|
| Arrival rate | 100% | 100% | — |
| **Signal efficiency** | **11%** | **100%** | **+89 pp** |
| Mean reward | 1442.6 | 1445.3 | +2.7 |

### What the Numbers Mean

**Signal efficiency is the core proof of learning.**

The untrained model (11% efficiency) toggles every signal it encounters, including ones already in the correct phase. It treats the action space as "clear everything" — a shallow shortcut.

After GRPO training (100% efficiency), the model learned to:
1. Read `current_phase` from the observation
2. Compute `needed_phase` based on the ambulance's direction of travel
3. Only send a `SignalControl` action when they differ

This is **not** a trivial pattern. The mapping between ambulance direction and required signal phase differs per intersection and must be reasoned about from the observation text each step.

The training curve shows the characteristic GRPO exploration-convergence pattern:
- **Iterations 1:** Model arrives (100% arrival) but wastes actions (11% efficiency)
- **Iterations 2–4:** Exploration phase — arrival drops to 0–25%, model tries aggressive strategies
- **Iterations 5–10:** Convergence — 100% arrival with 100% signal efficiency simultaneously

---

## Live Demo

**Demo video:** [Watch on YouTube](https://youtu.be/9O5z4IXXtcc)  
**Environment (OpenEnv WebSocket):** `wss://ajitg25-ambulance-green-corridor.hf.space/ws`  
**Visual simulation:** https://huggingface.co/spaces/Ajitg25/ambulance-green-corridor  
**GitHub (full code + notebook):** https://github.com/ajitg25/openEnv-hackathon/tree/main  
**Training notebook:** https://github.com/ajitg25/openEnv-hackathon/blob/main/examples/ambulance_grpo_training.ipynb

### Connecting Your Own Agent

```python
from ambulance_env import AmbulanceEnv, AmbulanceAction, SignalControl

async with AmbulanceEnv(base_url="https://ajitg25-ambulance-green-corridor.hf.space") as env:
    # Reset — get patient location, hospitals, initial state
    obs = (await env.reset()).observation

    # Step 1: Dispatch to specialist hospital
    obs = (await env.step(AmbulanceAction(hospital_id="hosp_b"))).observation

    # Step 2+: Clear only wrong-phase signals each step
    while not obs.done:
        controls = [
            SignalControl(
                row=s.row, col=s.col,
                phase="ns_green" if s.ambulance_direction in ("north", "south") else "ew_green"
            )
            for s in obs.lookahead_signals
            if s.phase != ("ns_green" if s.ambulance_direction in ("north", "south") else "ew_green")
        ]
        result = await env.step(AmbulanceAction(signal_controls=controls))
        obs = result.observation
```

### The Three Policies (Shown in Visual Demo)

| Policy | Behavior | Signal Efficiency | What It Demonstrates |
|---|---|---|---|
| No control | Does nothing | 0% | Pure baseline |
| Naive | Clears all signals | ~11% | Untrained LLM behavior |
| Smart | Clears only wrong-phase | 100% | Trained LLM behavior |

---

## Why This Matters

Emergency vehicle routing is deployed infrastructure. The gap between "clear one signal 300m ahead" and "reason about the full journey in a partially observable dynamic city" is exactly the gap this environment is designed to close.

Could a researcher write a paper about this? Yes:
> *"LLM-based adaptive emergency corridor planning under partial observability and dynamic constraints"*

That paper does not exist yet. This environment is the training ground for it.

---

*Built for the OpenEnv Hackathon India 2026 — Theme 3.1: World Modeling / Professional Tasks*
