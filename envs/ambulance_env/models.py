# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Ambulance Green Corridor Environment.

The agent acts as dispatcher, traffic signal manager, and dynamic re-router.
Traffic volume, road quality, and mid-episode events (accidents, closures, spikes)
force genuine reasoning — a simple rule-based system cannot solve this optimally.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv_core.env_server.types import Action, Observation, State  # type: ignore


class SignalControl(BaseModel):
    """A single traffic signal override instruction."""

    row: int = Field(..., description="Row of the intersection to control")
    col: int = Field(..., description="Column of the intersection to control")
    phase: str = Field(
        ...,
        description=(
            "'ns_green' = North-South green (East-West red), "
            "'ew_green' = East-West green (North-South red)"
        ),
    )


class HospitalInfo(BaseModel):
    """Static information about one hospital."""

    hospital_id: str
    name: str
    location: Tuple[int, int] = Field(..., description="(row, col) on the city grid")
    specialization: str = Field(
        ...,
        description="'general' | 'cardiac' | 'trauma' | 'stroke'",
    )
    at_capacity: bool = Field(False, description="True = cannot accept patients right now")
    distance_to_patient: int = Field(..., description="Manhattan distance from patient location")
    travel_time_estimate: float = Field(
        ..., description="Estimated travel time in simulated seconds"
    )


class SignalInfo(BaseModel):
    """Current state of one traffic signal in the lookahead window."""

    row: int
    col: int
    phase: str = Field(..., description="'ns_green' or 'ew_green'")
    seconds_until_change: float = Field(
        ..., description="Simulated seconds until the signal naturally changes phase"
    )
    traffic_density: float = Field(
        ..., description="Road congestion 0.0 (empty) to 1.0 (gridlocked)"
    )
    ambulance_direction: str = Field(
        ...,
        description="Direction the ambulance will travel through this intersection: north/south/east/west",
    )


class RoadSegment(BaseModel):
    """Describes one road segment between two adjacent intersections."""

    from_pos: Tuple[int, int] = Field(..., description="Start intersection (row, col)")
    to_pos: Tuple[int, int] = Field(..., description="End intersection (row, col)")
    direction: str = Field(..., description="Direction of travel: north|south|east|west")
    road_type: str = Field(..., description="highway|main|residential|damaged")
    road_quality: float = Field(..., description="0.0 (worst) to 1.0 (perfect)")
    traffic_volume: float = Field(..., description="0.0 (empty) to 1.0 (gridlocked)")
    blocked: bool = Field(False, description="True if an event has made this segment impassable")
    estimated_transit_time: float = Field(
        ..., description="Seconds to cross at current traffic + quality + signal conditions"
    )


class DynamicEvent(BaseModel):
    """A mid-episode event that may force re-routing."""

    event_type: str = Field(..., description="accident|traffic_spike|road_closure")
    position: Tuple[int, int] = Field(..., description="Intersection where the event occurred")
    severity: float = Field(..., description="0.0 (minor) to 1.0 (severe)")
    description: str = Field(..., description="Human-readable event summary")


class RouteOption(BaseModel):
    """A complete route from current ambulance location to a hospital."""

    hospital_id: str
    hospital_name: str
    path: List[Tuple[int, int]] = Field(..., description="List of intersections along the route")
    segments: List[RoadSegment] = Field(..., description="Road segments along the path")
    estimated_time: float = Field(..., description="Total seconds at current conditions")
    num_damaged_segments: int = Field(..., description="Segments with road_type=='damaged'")
    num_heavy_traffic_segments: int = Field(
        ..., description="Segments with traffic_volume > 0.7"
    )


class AmbulanceAction(Action):
    """
    Agent action — valid at every step after dispatch.

    hospital_id:         Switch destination to a different hospital (triggers full re-route).
    signal_controls:     Override up to 3 signals in the lookahead window.
    preferred_direction: Hint the routing engine to prefer a specific turn at the
                         current intersection (north|south|east|west).
    """

    hospital_id: Optional[str] = Field(
        None,
        description="ID of hospital to dispatch/re-route to. Valid at any step.",
    )
    signal_controls: List[SignalControl] = Field(
        default_factory=list,
        description="Signal overrides for up to 3 intersections ahead",
    )
    preferred_direction: Optional[str] = Field(
        None,
        description="Force a turn at the current intersection: north|south|east|west",
    )


class AmbulanceObservation(Observation):
    """Full observation returned after every reset/step."""

    # --- Patient ---
    patient_location: Tuple[int, int] = Field(..., description="Patient's grid position (row, col)")
    patient_condition: str = Field(
        ..., description="Medical emergency type: 'cardiac' | 'trauma' | 'stroke'"
    )

    # --- Ambulance ---
    ambulance_location: Tuple[int, int] = Field(
        ..., description="Ambulance's current intersection (row, col)"
    )
    current_segment: Optional[RoadSegment] = Field(
        None, description="Road segment the ambulance is currently traversing"
    )

    # --- Route ---
    current_route: RouteOption = Field(..., description="Current planned route with segment details")
    alternative_routes: List[RouteOption] = Field(
        default_factory=list, description="Up to 2 alternative routes the agent may prefer"
    )
    target_hospital_id: Optional[str] = Field(None, description="Currently targeted hospital ID")
    intersections_remaining: int = Field(0, description="Intersections left on current route")

    # --- Lookahead signals (next 3 intersections) ---
    lookahead_signals: List[SignalInfo] = Field(
        default_factory=list,
        description="Signal state for the next 1-3 intersections the ambulance will cross",
    )

    # --- Dynamic events ---
    active_events: List[DynamicEvent] = Field(
        default_factory=list,
        description="Dynamic events currently active on the map",
    )

    # --- Hospitals ---
    hospitals: List[HospitalInfo] = Field(
        default_factory=list, description="All hospitals with distance and specialization info"
    )

    # --- Timing + metrics ---
    time_elapsed_seconds: float = Field(0.0, description="Simulated seconds since dispatch")
    time_limit_seconds: float = Field(300.0, description="Episode time limit in simulated seconds")
    last_speed_factor: float = Field(
        1.0, description="Speed factor on last step: 1.0=full, 0.0=blocked"
    )
    stops_at_red: int = Field(0, description="Steps where the ambulance was stopped by a red signal")
    total_distance_covered: float = Field(0.0, description="Intersections passed so far")

    # --- Signal efficiency ---
    necessary_toggles: int = Field(0, description="Signals correctly changed from wrong to right phase")
    unnecessary_toggles: int = Field(0, description="Signals toggled that were already correct")
    first_signal_failures: int = Field(
        0, description="Times the next intersection was red when the ambulance arrived"
    )
    signal_efficiency: float = Field(
        1.0, description="necessary / (necessary + unnecessary) toggles ratio"
    )

    # --- Re-routing metrics ---
    successful_reroutes: int = Field(
        0, description="Times the agent successfully avoided a blocked/slow route"
    )
    damaged_segments_traversed: int = Field(
        0, description="Damaged road segments crossed (patient comfort metric)"
    )

    done: bool = Field(False)
    reward: Optional[float] = Field(None)


class AmbulanceState(State):
    """Episode-level metadata and cumulative statistics."""

    difficulty: str = Field("easy", description="'easy' | 'medium' | 'hard'")
    patient_condition: str = Field("cardiac")
    target_hospital_id: Optional[str] = Field(None)
    hospital_matched_condition: bool = Field(False)
    total_stops: int = Field(0)
    arrival_time: Optional[float] = Field(None)
    success: bool = Field(False)
    necessary_toggles: int = Field(0)
    unnecessary_toggles: int = Field(0)
    first_signal_failures: int = Field(0)
    signal_efficiency: float = Field(1.0)
    successful_reroutes: int = Field(0)
    damaged_segments_traversed: int = Field(0)
