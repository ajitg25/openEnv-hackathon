"""
Ambulance Green Corridor — Showcase Server
===========================================
Mounts the real OpenEnv app (with /ws, /reset, /step, /state, /schema, /mcp)
and adds the visual frontend + extra demo endpoints on top.

    AMBULANCE_DIFFICULTY=easy python showcase.py

Open http://localhost:7860 for the visual demo.
The validator can connect to ws://localhost:7860/ws as usual.
"""

import os
import random
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

from ambulance_env.server.ambulance_environment import AmbulanceEnvironment, _seg_key
from ambulance_env.models import AmbulanceAction, AmbulanceObservation, SignalControl

# ---------------------------------------------------------------------------
# The real OpenEnv app — has /ws, /reset, /step, /state, /schema, /health, /mcp
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server.http_server import create_app as create_openenv_app
except ImportError:
    create_openenv_app = None

_difficulty = os.getenv("AMBULANCE_DIFFICULTY", "easy")
FRONTEND_DIR = Path(__file__).parent / "frontend" / "public"


def _create_env() -> AmbulanceEnvironment:
    return AmbulanceEnvironment(difficulty=_difficulty)


if create_openenv_app:
    # Build the real OpenEnv app with all endpoints including WebSocket
    app = create_openenv_app(
        _create_env,
        AmbulanceAction,
        AmbulanceObservation,
        env_name="ambulance_env",
        max_concurrent_envs=4,
    )
else:
    # Fallback if openenv not installed
    app = FastAPI(title="Ambulance Green Corridor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Stateful env instance for the frontend demo
# (separate from the OpenEnv per-session envs used by the validator)
# ---------------------------------------------------------------------------

_demo_env: Optional[AmbulanceEnvironment] = None


class DemoResetBody(BaseModel):
    difficulty: Optional[str] = None

class DemoStepBody(BaseModel):
    hospital_id: Optional[str] = None
    signal_controls: list = []
    preferred_direction: Optional[str] = None


@app.post("/api/reset")
def demo_reset(body: DemoResetBody = None):
    global _demo_env
    diff = (body and body.difficulty) or _difficulty
    if diff not in ("easy", "medium", "hard"):
        diff = "easy"
    _demo_env = AmbulanceEnvironment(difficulty=diff)
    obs = _demo_env.reset()
    return {"observation": obs.model_dump()}


@app.post("/api/step")
def demo_step(body: DemoStepBody):
    if _demo_env is None:
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
    obs = _demo_env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.get("/api/state")
def demo_state():
    if _demo_env is None:
        return {"error": "No active demo env"}
    return _demo_env.state.model_dump()


@app.get("/api/roads")
def demo_roads():
    if _demo_env is None:
        return {"error": "Call /api/reset first"}
    result = []
    for seg_key, seg in _demo_env._segments.items():
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
def demo_signals():
    if _demo_env is None:
        return {"error": "Call /api/reset first"}
    result = {}
    for (r, c), phase in _demo_env._signals.items():
        result[f"{r},{c}"] = phase
    return {"signals": result}


@app.post("/api/trigger_event")
def demo_trigger_event():
    if _demo_env is None:
        return {"error": "Call /api/reset first"}
    if not _demo_env._route or _demo_env._route_idx >= len(_demo_env._route) - 2:
        return {"error": "No upcoming route to block"}

    route = _demo_env._route
    idx = _demo_env._route_idx
    candidates = []
    for i in range(idx + 3, min(idx + 6, len(route) - 1)):
        key = _seg_key(route[i], route[i + 1])
        seg = _demo_env._segments.get(key)
        if seg and not seg.blocked:
            candidates.append((key, seg, route[i], route[i + 1]))

    if not candidates:
        for i in range(idx + 1, min(idx + 4, len(route) - 1)):
            key = _seg_key(route[i], route[i + 1])
            seg = _demo_env._segments.get(key)
            if seg and not seg.blocked:
                candidates.append((key, seg, route[i], route[i + 1]))

    if not candidates:
        return {"error": "No available segment to block"}

    key, seg, pos_a, pos_b = random.choice(candidates)
    event_type = random.choice(["accident", "road_closure"])
    seg.blocked = True
    desc = f"{'Accident' if event_type == 'accident' else 'Road closure'} between {pos_a} and {pos_b}!"

    for sk, s in _demo_env._segments.items():
        if not s.blocked:
            for p in list(sk):
                p = p if isinstance(p, tuple) else tuple(p)
                if abs(p[0] - pos_a[0]) <= 1 and abs(p[1] - pos_a[1]) <= 1:
                    s.current_traffic = min(1.0, s.current_traffic + 0.3)
                    break

    return {"event": event_type, "blocked": [list(pos_a), list(pos_b)], "description": desc}


@app.post("/api/spike_traffic")
def demo_spike_traffic():
    if _demo_env is None:
        return {"error": "Call /api/reset first"}
    spiked = 0
    for key, seg in _demo_env._segments.items():
        if not seg.blocked and random.random() < 0.3:
            seg.current_traffic = min(1.0, seg.current_traffic + random.uniform(0.2, 0.5))
            spiked += 1
    return {"spiked": spiked, "total": len(_demo_env._segments)}


@app.get("/api/health")
def demo_health():
    return {"status": "ok", "difficulty": _difficulty}


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

@app.get("/web")
async def serve_web():
    return FileResponse(FRONTEND_DIR / "index.html")

app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    print(f"\n  Ambulance Green Corridor")
    print(f"  Visual demo : http://localhost:{port}/web")
    print(f"  OpenEnv API : http://localhost:{port}/docs")
    print(f"  WebSocket   : ws://localhost:{port}/ws")
    print(f"  Difficulty  : {_difficulty}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
