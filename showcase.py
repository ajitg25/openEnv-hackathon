"""
Ambulance Green Corridor — Showcase Server
===========================================
Single server that serves the visual frontend AND the environment API.
One command to run the entire demo:

    AMBULANCE_DIFFICULTY=easy python showcase.py

Then open http://localhost:7860
"""

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Ensure envs/ is importable
sys.path.insert(0, str(Path(__file__).parent / "envs"))

import random

from ambulance_env.server.ambulance_environment import AmbulanceEnvironment, _seg_key
from ambulance_env.models import AmbulanceAction, SignalControl

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Ambulance Green Corridor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: Optional[AmbulanceEnvironment] = None
_difficulty = os.getenv("AMBULANCE_DIFFICULTY", "easy")

FRONTEND_DIR = Path(__file__).parent / "frontend" / "public"


# ---------------------------------------------------------------------------
# Environment API (under /api prefix)
# ---------------------------------------------------------------------------

class StepBody(BaseModel):
    hospital_id: Optional[str] = None
    signal_controls: list = []
    preferred_direction: Optional[str] = None


@app.post("/api/reset")
def reset():
    global _env
    _env = AmbulanceEnvironment(difficulty=_difficulty)
    obs = _env.reset()
    return {"observation": obs.model_dump()}


@app.post("/api/step")
def step(body: StepBody):
    if _env is None:
        return {"error": "Call /api/reset first"}
    controls = [
        SignalControl(row=c["row"], col=c["col"], phase=c["phase"])
        for c in body.signal_controls
        if isinstance(c, dict) and "row" in c
    ]
    action = AmbulanceAction(
        hospital_id=body.hospital_id,
        signal_controls=controls,
        preferred_direction=body.preferred_direction,
    )
    obs = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.get("/api/state")
def state():
    if _env is None:
        return {"error": "No active env"}
    return _env.state.model_dump()


@app.get("/api/roads")
def roads():
    if _env is None:
        return {"error": "Call /api/reset first"}
    result = []
    for seg_key, seg in _env._segments.items():
        pos_list = list(seg_key)
        a = pos_list[0] if isinstance(pos_list[0], tuple) else tuple(pos_list[0])
        b = pos_list[1] if isinstance(pos_list[1], tuple) else tuple(pos_list[1])
        result.append({
            "from": list(a), "to": list(b),
            "type": seg.road_type,
            "traffic": round(seg.current_traffic, 3),
            "quality": round(seg.quality, 3),
            "blocked": seg.blocked,
        })
    return {"roads": result}


@app.get("/api/signals")
def signals():
    if _env is None:
        return {"error": "Call /api/reset first"}
    result = {}
    for (r, c), phase in _env._signals.items():
        result[f"{r},{c}"] = phase
    return {"signals": result}


@app.post("/api/trigger_event")
def trigger_event():
    """Force an obstacle on or near the ambulance's upcoming route."""
    if _env is None:
        return {"error": "Call /api/reset first"}
    if not _env._route or _env._route_idx >= len(_env._route) - 2:
        return {"error": "No upcoming route to block"}

    # Pick a segment 3-5 steps ahead on the route (not the immediate next one)
    route = _env._route
    idx = _env._route_idx
    candidates = []
    for i in range(idx + 3, min(idx + 6, len(route) - 1)):
        key = _seg_key(route[i], route[i + 1])
        seg = _env._segments.get(key)
        if seg and not seg.blocked:
            candidates.append((key, seg, route[i], route[i + 1]))

    # If nothing ahead, try any unblocked segment near the route
    if not candidates:
        for i in range(idx + 1, min(idx + 4, len(route) - 1)):
            key = _seg_key(route[i], route[i + 1])
            seg = _env._segments.get(key)
            if seg and not seg.blocked:
                candidates.append((key, seg, route[i], route[i + 1]))

    if not candidates:
        return {"error": "No available segment to block"}

    key, seg, pos_a, pos_b = random.choice(candidates)
    event_type = random.choice(["accident", "road_closure"])

    if event_type == "accident":
        seg.blocked = True
        desc = f"Accident between {pos_a} and {pos_b} — road blocked!"
    else:
        seg.blocked = True
        desc = f"Road closure between {pos_a} and {pos_b} — construction!"

    # Also spike traffic on nearby segments
    for sk, s in _env._segments.items():
        if not s.blocked:
            pos_list = list(sk)
            for p in pos_list:
                p = p if isinstance(p, tuple) else tuple(p)
                if abs(p[0] - pos_a[0]) <= 1 and abs(p[1] - pos_a[1]) <= 1:
                    s.current_traffic = min(1.0, s.current_traffic + 0.3)
                    break

    return {
        "event": event_type,
        "blocked": [list(pos_a), list(pos_b)],
        "description": desc,
    }


@app.post("/api/spike_traffic")
def spike_traffic():
    """Spike traffic volume across random segments to simulate congestion."""
    if _env is None:
        return {"error": "Call /api/reset first"}
    spiked = 0
    for key, seg in _env._segments.items():
        if not seg.blocked and random.random() < 0.3:
            seg.current_traffic = min(1.0, seg.current_traffic + random.uniform(0.2, 0.5))
            spiked += 1
    return {"spiked": spiked, "total": len(_env._segments)}


@app.get("/api/health")
def health():
    return {"status": "ok", "difficulty": _difficulty}


# ---------------------------------------------------------------------------
# OpenEnv-compatible endpoints (no /api prefix — for the validator)
# The validator expects POST /reset, POST /step, GET /state at the root.
# These share the same _env instance as the /api/* routes.
# ---------------------------------------------------------------------------

class ValidatorStepBody(BaseModel):
    action: Optional[dict] = None

@app.post("/reset")
def validator_reset():
    global _env
    _env = AmbulanceEnvironment(difficulty=_difficulty)
    obs = _env.reset()
    return {"observation": obs.model_dump()}

@app.post("/step")
def validator_step(body: ValidatorStepBody):
    if _env is None:
        return {"error": "Call /reset first"}
    action_data = body.action or {}
    controls = [
        SignalControl(row=c["row"], col=c["col"], phase=c["phase"])
        for c in action_data.get("signal_controls", [])
        if isinstance(c, dict) and "row" in c
    ]
    action = AmbulanceAction(
        hospital_id=action_data.get("hospital_id"),
        signal_controls=controls,
        preferred_direction=action_data.get("preferred_direction"),
    )
    obs = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }

@app.get("/state")
def validator_state():
    if _env is None:
        return {"error": "No active env"}
    return {"state": _env.state.model_dump()}

@app.get("/health")
def validator_health():
    return {"status": "ok", "difficulty": _difficulty}


# ---------------------------------------------------------------------------
# Serve frontend static files
# ---------------------------------------------------------------------------

@app.get("/web")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")

# Also serve at root for convenience
@app.get("/")
async def serve_root():
    return FileResponse(FRONTEND_DIR / "index.html")

app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    print(f"\n  🚑 Ambulance Green Corridor")
    print(f"  Open http://localhost:{port}")
    print(f"  Difficulty: {_difficulty}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
