# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Ambulance Green Corridor Environment – WebSocket client.

Typical usage
-------------
    from envs.ambulance_env import AmbulanceEnv, AmbulanceAction, SignalControl

    with AmbulanceEnv(base_url="http://localhost:8000") as env:
        obs = env.reset().observation

        # Phase 1: dispatch
        result = env.step(AmbulanceAction(hospital_id="hosp_b"))
        obs = result.observation  # phase == 'routing'

        # Phase 2: route with green corridor
        while not result.done:
            controls = []
            for sig in obs.lookahead_signals:
                # Give the ambulance a green light for its direction
                needed = "ns_green" if sig.ambulance_direction in ("north", "south") else "ew_green"
                if sig.phase != needed:
                    controls.append(SignalControl(row=sig.row, col=sig.col, phase=needed))
            result = env.step(AmbulanceAction(signal_controls=controls))
            obs = result.observation
"""

from __future__ import annotations

from typing import Any, Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    AmbulanceAction,
    AmbulanceObservation,
    AmbulanceState,
    HospitalInfo,
    SignalInfo,
)


class AmbulanceEnv(EnvClient[AmbulanceAction, AmbulanceObservation, AmbulanceState]):
    """WebSocket client for the Ambulance Green Corridor environment."""

    def _step_payload(self, action: AmbulanceAction) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if action.hospital_id is not None:
            payload["hospital_id"] = action.hospital_id
        payload["signal_controls"] = [
            {"row": sc.row, "col": sc.col, "phase": sc.phase}
            for sc in action.signal_controls
        ]
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AmbulanceObservation]:
        obs_data = payload.get("observation", {})

        hospitals: List[HospitalInfo] = [
            HospitalInfo(
                hospital_id=h["hospital_id"],
                name=h["name"],
                location=tuple(h["location"]),  # type: ignore[arg-type]
                specialization=h["specialization"],
                at_capacity=h.get("at_capacity", False),
                distance_to_patient=h["distance_to_patient"],
                travel_time_estimate=h["travel_time_estimate"],
            )
            for h in obs_data.get("hospitals", [])
        ]

        signals: List[SignalInfo] = [
            SignalInfo(
                row=s["row"],
                col=s["col"],
                phase=s["phase"],
                seconds_until_change=s["seconds_until_change"],
                traffic_density=s["traffic_density"],
                ambulance_direction=s["ambulance_direction"],
            )
            for s in obs_data.get("lookahead_signals", [])
        ]

        obs = AmbulanceObservation(
            patient_location=tuple(obs_data.get("patient_location", (0, 0))),  # type: ignore[arg-type]
            patient_condition=obs_data.get("patient_condition", "cardiac"),
            phase=obs_data.get("phase", "dispatch"),
            ambulance_location=tuple(obs_data.get("ambulance_location", (0, 0))),  # type: ignore[arg-type]
            route_to_hospital=[tuple(p) for p in obs_data.get("route_to_hospital", [])],  # type: ignore[misc]
            intersections_remaining=obs_data.get("intersections_remaining", 0),
            hospitals=hospitals,
            target_hospital_id=obs_data.get("target_hospital_id"),
            lookahead_signals=signals,
            time_elapsed_seconds=obs_data.get("time_elapsed_seconds", 0.0),
            time_limit_seconds=obs_data.get("time_limit_seconds", 300.0),
            last_speed_factor=obs_data.get("last_speed_factor", 1.0),
            stops_at_red=obs_data.get("stops_at_red", 0),
            total_distance_covered=obs_data.get("total_distance_covered", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> AmbulanceState:
        return AmbulanceState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            difficulty=payload.get("difficulty", "easy"),
            patient_condition=payload.get("patient_condition", "cardiac"),
            target_hospital_id=payload.get("target_hospital_id"),
            hospital_matched_condition=payload.get("hospital_matched_condition", False),
            total_stops=payload.get("total_stops", 0),
            arrival_time=payload.get("arrival_time"),
            success=payload.get("success", False),
        )
