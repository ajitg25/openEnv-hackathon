"""
Thin stateful API server for the JS frontend.
Holds one environment instance in memory so /step persists state across calls.

Usage:
    PYTHONPATH=../envs python env_server.py
    # or: PYTHONPATH=../envs uvicorn env_server:app --port 8000 --reload
"""

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add envs to path
sys.path.insert(0, str(Path(__file__).parent.parent / "envs"))

from ambulance_env.server.ambulance_environment import AmbulanceEnvironment, _seg_key
from ambulance_env.models import AmbulanceAction, SignalControl

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_env: Optional[AmbulanceEnvironment] = None
_difficulty = os.getenv("AMBULANCE_DIFFICULTY", "easy")


class StepBody(BaseModel):
    hospital_id: Optional[str] = None
    signal_controls: list = []
    preferred_direction: Optional[str] = None


@app.post("/reset")
def reset():
    global _env
    _env = AmbulanceEnvironment(difficulty=_difficulty)
    obs = _env.reset()
    return {"observation": obs.model_dump()}


@app.post("/step")
def step(body: StepBody):
    if _env is None:
        return {"error": "Call /reset first"}

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


@app.get("/state")
def state():
    if _env is None:
        return {"error": "No active env"}
    return _env.state.model_dump()


@app.get("/roads")
def roads():
    """All road segments with current traffic/quality/blocked state."""
    if _env is None:
        return {"error": "Call /reset first"}
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


@app.get("/signals")
def signals():
    """All traffic signal states."""
    if _env is None:
        return {"error": "Call /reset first"}
    result = {}
    for (r, c), phase in _env._signals.items():
        result[f"{r},{c}"] = phase
    return {"signals": result}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"\n  Env server on http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
