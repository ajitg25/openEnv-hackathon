---
title: Ambulance Green Corridor
emoji: 🚑
colorFrom: red
colorTo: green
sdk: docker
app_port: 7860
---

# Ambulance Green Corridor — OpenEnv Hackathon 2026

> **Can an LLM learn to save lives by managing city traffic?**
>
> Every minute of delay in cardiac arrest reduces survival by 10%.
> This environment trains an LLM agent to dispatch ambulances to the right
> hospital and clear a rolling green corridor through live city traffic.

---

## Problem

Emergency vehicle routing is a real, unsolved AI challenge. Current systems
use fixed preemption rules. An LLM agent that reasons about traffic state,
hospital specialization, and signal timing can do far better — and this
environment proves it.

---

## Environment

The agent plays two roles each episode:

**1. Dispatcher** — given a patient's location and condition (cardiac / trauma /
stroke), choose the best hospital. Specialist hospitals score higher but may be
farther away. One hospital may be at capacity.

**2. Traffic Signal Manager** — as the ambulance moves, the agent sees a rolling
3-signal lookahead window. It must clear only the signals that are in the wrong
phase for the ambulance's direction. Toggling already-green signals wastes
actions and costs reward.

### Observation (per step)
| Field | Description |
|---|---|
| `patient_condition` | cardiac / trauma / stroke |
| `phase` | dispatch or routing |
| `ambulance_location` | current grid cell |
| `lookahead_signals` | next 1–3 intersections: phase, direction, density |
| `hospitals` | all hospitals with distance, specialization, capacity |
| `signal_efficiency` | necessary_toggles / total_toggles (live metric) |

### Action
```json
{
  "hospital_id": "hosp_b",
  "signal_controls": [
    {"row": 3, "col": 0, "phase": "ns_green"}
  ]
}
```

### Reward
- `+1000` base arrival bonus
- `+time_bonus` faster arrival = more reward (up to +500)
- `+300` specialist hospital matched to patient condition
- `−20` per red-light stop
- `−2` per unnecessary signal toggle (step penalty)
- `−5` per unnecessary toggle (arrival penalty)
- `−500` episode timeout

### Difficulty Levels
| Level | Grid | Hospitals | Traffic | Time Limit |
|---|---|---|---|---|
| easy | 6×6 | 2 | Low | 200s |
| medium | 8×8 | 3 | Moderate | 300s |
| hard | 12×12 | 5 (1 at capacity) | Heavy | 400s |

---

## Results

| Policy | Arrival time | Red stops | Signal efficiency | Reward |
|---|---|---|---|---|
| No signal control (baseline) | ~140s | 24 | — | ~987 |
| Naive (toggle everything) | ~40s | 0 | 12% | ~1629 |
| Smart (toggle only wrong-phase) | ~40s | 0 | 100% | ~1732 |

> Training plots will be added after the Colab training run.

---

## Training

Training script: [`examples/ambulance_grpo_training.py`](examples/ambulance_grpo_training.py)

- Model: Qwen2.5-0.5B-Instruct (via Unsloth, 4-bit, fits free T4)
- Algorithm: GRPO (TRL)
- Runtime: ~30–45 min on Colab free tier

---

## Setup

```bash
uv sync
AMBULANCE_DIFFICULTY=easy uvicorn ambulance_env.server.app:app \
  --app-dir envs --host 0.0.0.0 --port 7860
```

---

## Links

- HF Space: _coming soon_
- Training notebook: _coming soon_
- Blog post / video: _coming soon_
