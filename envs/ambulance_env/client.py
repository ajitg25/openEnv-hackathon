# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Ambulance Green Corridor Environment – WebSocket client.

The client parses server JSON into typed Pydantic models so the training
script can work with structured observations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    AmbulanceAction,
    AmbulanceObservation,
    AmbulanceState,
    DynamicEvent,
    HospitalInfo,
    RoadSegment,
    RouteOption,
    SignalControl,
    SignalInfo,
)


def _parse_tuple(val) -> tuple:
    if isinstance(val, (list, tuple)):
        return tuple(val)
    return (0, 0)


def _parse_hospitals(raw: list) -> List[HospitalInfo]:
    return [
        HospitalInfo(
            hospital_id=h["hospital_id"],
            name=h["name"],
            location=_parse_tuple(h["location"]),  # type: ignore[arg-type]
            specialization=h["specialization"],
            at_capacity=h.get("at_capacity", False),
            distance_to_patient=h["distance_to_patient"],
            travel_time_estimate=h["travel_time_estimate"],
        )
        for h in raw
    ]


def _parse_signals(raw: list) -> List[SignalInfo]:
    return [
        SignalInfo(
            row=s["row"],
            col=s["col"],
            phase=s["phase"],
            seconds_until_change=s["seconds_until_change"],
            traffic_density=s["traffic_density"],
            ambulance_direction=s["ambulance_direction"],
        )
        for s in raw
    ]


def _parse_segment(raw: dict) -> RoadSegment:
    return RoadSegment(
        from_pos=_parse_tuple(raw["from_pos"]),  # type: ignore[arg-type]
        to_pos=_parse_tuple(raw["to_pos"]),  # type: ignore[arg-type]
        direction=raw["direction"],
        road_type=raw["road_type"],
        road_quality=raw["road_quality"],
        traffic_volume=raw["traffic_volume"],
        blocked=raw.get("blocked", False),
        estimated_transit_time=raw["estimated_transit_time"],
    )


def _parse_route(raw: dict) -> RouteOption:
    return RouteOption(
        hospital_id=raw["hospital_id"],
        hospital_name=raw["hospital_name"],
        path=[_parse_tuple(p) for p in raw.get("path", [])],  # type: ignore[misc]
        segments=[_parse_segment(s) for s in raw.get("segments", [])],
        estimated_time=raw["estimated_time"],
        num_damaged_segments=raw.get("num_damaged_segments", 0),
        num_heavy_traffic_segments=raw.get("num_heavy_traffic_segments", 0),
    )


def _parse_events(raw: list) -> List[DynamicEvent]:
    return [
        DynamicEvent(
            event_type=e["event_type"],
            position=_parse_tuple(e["position"]),  # type: ignore[arg-type]
            severity=e["severity"],
            description=e["description"],
        )
        for e in raw
    ]


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
        if action.preferred_direction is not None:
            payload["preferred_direction"] = action.preferred_direction
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AmbulanceObservation]:
        obs_data = payload.get("observation", {})

        # Parse current_route
        route_raw = obs_data.get("current_route", {})
        current_route = _parse_route(route_raw) if route_raw else RouteOption(
            hospital_id="", hospital_name="", path=[], segments=[],
            estimated_time=0.0, num_damaged_segments=0, num_heavy_traffic_segments=0,
        )

        # Parse alternatives
        alternatives = [_parse_route(r) for r in obs_data.get("alternative_routes", [])]

        # Parse current_segment
        seg_raw = obs_data.get("current_segment")
        current_segment = _parse_segment(seg_raw) if seg_raw else None

        obs = AmbulanceObservation(
            patient_location=_parse_tuple(obs_data.get("patient_location", (0, 0))),  # type: ignore[arg-type]
            patient_condition=obs_data.get("patient_condition", "cardiac"),
            ambulance_location=_parse_tuple(obs_data.get("ambulance_location", (0, 0))),  # type: ignore[arg-type]
            current_segment=current_segment,
            current_route=current_route,
            alternative_routes=alternatives,
            target_hospital_id=obs_data.get("target_hospital_id"),
            intersections_remaining=obs_data.get("intersections_remaining", 0),
            lookahead_signals=_parse_signals(obs_data.get("lookahead_signals", [])),
            active_events=_parse_events(obs_data.get("active_events", [])),
            hospitals=_parse_hospitals(obs_data.get("hospitals", [])),
            time_elapsed_seconds=obs_data.get("time_elapsed_seconds", 0.0),
            time_limit_seconds=obs_data.get("time_limit_seconds", 300.0),
            last_speed_factor=obs_data.get("last_speed_factor", 1.0),
            stops_at_red=obs_data.get("stops_at_red", 0),
            total_distance_covered=obs_data.get("total_distance_covered", 0.0),
            necessary_toggles=obs_data.get("necessary_toggles", 0),
            unnecessary_toggles=obs_data.get("unnecessary_toggles", 0),
            first_signal_failures=obs_data.get("first_signal_failures", 0),
            signal_efficiency=obs_data.get("signal_efficiency", 1.0),
            successful_reroutes=obs_data.get("successful_reroutes", 0),
            damaged_segments_traversed=obs_data.get("damaged_segments_traversed", 0),
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
            necessary_toggles=payload.get("necessary_toggles", 0),
            unnecessary_toggles=payload.get("unnecessary_toggles", 0),
            first_signal_failures=payload.get("first_signal_failures", 0),
            signal_efficiency=payload.get("signal_efficiency", 1.0),
            successful_reroutes=payload.get("successful_reroutes", 0),
            damaged_segments_traversed=payload.get("damaged_segments_traversed", 0),
        )
