# Can an LLM Learn to Save Lives by Managing City Traffic?

**tl;dr:** We built an OpenEnv environment that trains an LLM to act as emergency dispatcher + city traffic signal manager. After GRPO training, signal efficiency jumped from **11% → 100%**. Here's how it works and why it genuinely needs an LLM — not just a rule.

---

## The Problem

In a cardiac emergency, every minute of delay costs ~10% survival probability.

Existing GPS-based emergency preemption systems (like Opticom) clear one traffic signal when an ambulance is 300m away. That's reactive, single-intersection, and has no awareness of what lies ahead.

Our environment asks: **can an LLM reason about the full journey — hospital selection, road quality, live traffic, and dynamic events — to get the ambulance there faster?**

---

## Why This Needs an LLM (Not a Rule)

Consider this scenario:

- **Hospital A:** 6 intersections away, but 3 road segments are gridlocked. Clearing signals helps, but heavy traffic means the ambulance crawls at ~20% speed even on green.
- **Hospital B:** 8 intersections, lighter traffic, highway-quality roads. ETA is actually 40 seconds faster.
- **Midway:** an accident blocks the planned route. The system must re-route in real time.

No rule-based system can solve this. The agent must simultaneously reason about:

- Distance vs. traffic volume vs. road quality
- Hospital specialization (cardiac patient → cardiac centre, not general hospital)
- Dynamic events appearing mid-journey (accidents, road closures, traffic spikes)
- Which signals actually need clearing — toggling an already-green signal wastes an action and costs reward

---

## The Environment

Built on **[OpenEnv](https://github.com/meta-pytorch/OpenEnv)** — the hackathon framework for LLM training environments.

### What the agent sees each step

```
=== EMERGENCY DISPATCH ===
Patient  : (6, 3) | condition: cardiac
Ambulance: (6, 4) | time: 40s / 300s

⚠ DYNAMIC EVENTS:
  [ACCIDENT] at (4,3) — blocking road (severity=0.8)

CURRENT ROUTE → hosp_a
  ETA=251s | segments=8 | damaged=2 | heavy_traffic=1
  (6,4)→(5,4) | residential | quality=moderate | traffic=45% | est=22s
  (5,4)→(4,4) | damaged     | quality=POTHOLED | traffic=62% | est=41s [BLOCKED]

ALTERNATIVES (consider switching if ETA much lower):
  hosp_c (cardiac) <- specialist match: ETA=130s | damaged=0 | heavy=0

HOSPITALS:
  hosp_a: City General | spec=general | est=251s
  hosp_c: Cardiac Centre | spec=cardiac | est=130s <- specialist match

SIGNALS (only change WRONG ones):
  (5,4): ns_green | dir=north | OK
  (4,4): ew_green | dir=north | WRONG — needs ns_green

ACTION: {"hospital_id": "hosp_c", "signal_controls": [{"row": 4, "col": 4, "phase": "ns_green"}], "preferred_direction": null}
```

### Reward function — designed to be hard to game

| Component | Value |
|---|---|
| Arrival bonus | +1000 |
| Time bonus | up to +500 (faster = more) |
| Specialist hospital match | +300 |
| Red light stop | −20 each |
| **Unnecessary signal toggle** | **−2/−5 each** |
| Damaged road segments traversed | −10 each |
| Successful re-route | +50 each |

The unnecessary toggle penalty is the key design decision. An agent that blindly clears every signal it sees scores *worse* than one that reads the signal state first. This forces the LLM to actually reason about observations rather than pattern-match to a fixed action.

### Difficulty levels

| Level | Grid | Hospitals | Traffic | Dynamic Events | Time Limit |
|---|---|---|---|---|---|
| easy | 6×6 | 2 | Low | 5%/step | 200s |
| medium | 8×8 | 3 | Moderate | 10%/step | 300s |
| hard | 12×12 | 5 (1 at capacity) | Heavy | 15%/step | 400s |

---

## Training

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` + LoRA (r=16, 2.1M trainable params)
- **Algorithm:** GRPO (Group Relative Policy Optimisation via HuggingFace TRL)
- **Setup:** 10 iterations × 4 episodes per iteration
- **Environment:** Live OpenEnv server running alongside training loop

![Training curves](ambulance_training_results.png)
*Four panels: Episode reward, Hospital arrival rate, Signal efficiency (11%→100%), Adaptive re-routing*

### Results

| Metric | Baseline (untrained) | Trained | Change |
|---|---|---|---|
| Arrival rate | 100% | 100% | — |
| **Signal efficiency** | **11%** | **100%** | **+89 percentage points** |
| Mean reward | 1442.6 | 1445.3 | +2.7 |
| Mean travel time | 125s | 127.5s | — |

### What the numbers mean

**Signal efficiency is the headline metric.** The untrained model toggled every signal it saw — including ones already in the correct phase — scoring unnecessary toggle penalties on every step. After GRPO training, the model learned to read `sig.phase` vs `sig.ambulance_direction` and only act when a signal genuinely needs changing.

The training curve shows characteristic GRPO behaviour:
- **Iterations 1:** model arrives but wastes actions (efficiency=11%)
- **Iterations 2–4:** exploration phase — model tries aggressive strategies, arrival drops to 0–25%
- **Iterations 5–10:** sharp convergence — 100% arrival, 100% signal efficiency, stable reward

This exploration→convergence pattern is the training story. A rule-based system would never show this curve — it would be flat from iteration 1.

---

## Why This Environment Matters

Emergency vehicle routing is a real, unsolved problem in smart city infrastructure. Current systems are:

- **Reactive:** clear one signal at a time, 300m in advance
- **Unaware of road quality:** a potholed road still gets treated as highway
- **Static:** no dynamic re-routing when accidents occur
- **Oblivious to hospital specialization:** nearest hospital isn't always right hospital

An LLM trained on this environment learns to reason about all four simultaneously. That's a capability that doesn't exist in any deployed system today.

Could a researcher write a paper about this? Yes — "LLM-based adaptive emergency corridor planning under partial observability and dynamic constraints" is a legitimate research direction this environment enables.

---

## Try It

**Live environment:** https://huggingface.co/spaces/Ajitg25/ambulance-green-corridor

**Code + training notebook:** https://github.com/ajitg25/openEnv-hackathon/tree/final

```python
from ambulance_env import AmbulanceEnv, AmbulanceAction, SignalControl

async with AmbulanceEnv(base_url="https://ajitg25-ambulance-green-corridor.hf.space") as env:
    obs = (await env.reset()).observation
    # Dispatch to specialist hospital
    obs = (await env.step(AmbulanceAction(hospital_id="hosp_b"))).observation
    # Clear only wrong-phase signals
    controls = [
        SignalControl(row=s.row, col=s.col,
                      phase="ns_green" if s.ambulance_direction in ("north","south") else "ew_green")
        for s in obs.lookahead_signals
        if s.phase != ("ns_green" if s.ambulance_direction in ("north","south") else "ew_green")
    ]
    result = await env.step(AmbulanceAction(signal_controls=controls))
```

---

---

*Built for the OpenEnv Hackathon India 2026.*
