# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Ambulance Green Corridor Environment.

The agent acts as both a dispatch controller (choosing the best hospital)
and a traffic signal manager (creating a rolling green corridor ahead of
the ambulance to minimize response time).
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


class AmbulanceAction(Action):
    """
    Agent action.

    Phase 'dispatch':  set ``hospital_id`` to choose where to send the ambulance.
    Phase 'routing':   set ``signal_controls`` (up to 3) to clear the path ahead.
    Both fields may be present in the same action; the environment ignores
    fields that are irrelevant to the current phase.
    """

    hospital_id: Optional[str] = Field(
        None,
        description="ID of the hospital to dispatch to (dispatch phase only)",
    )
    signal_controls: List[SignalControl] = Field(
        default_factory=list,
        description="Signal overrides for up to 3 intersections in the lookahead window",
    )


class AmbulanceObservation(Observation):
    """Full observation returned after every reset/step."""

    # --- Patient ---
    patient_location: Tuple[int, int] = Field(
        ..., description="Patient's grid position (row, col)"
    )
    patient_condition: str = Field(
        ..., description="Medical emergency type: 'cardiac' | 'trauma' | 'stroke'"
    )

    # --- Episode phase ---
    phase: str = Field(
        ...,
        description=(
            "'dispatch' = choose a hospital; "
            "'routing' = manage signals to clear the path"
        ),
    )

    # --- Ambulance ---
    ambulance_location: Tuple[int, int] = Field(
        ..., description="Ambulance's current intersection (row, col)"
    )
    route_to_hospital: List[Tuple[int, int]] = Field(
        default_factory=list,
        description="Remaining intersections on the planned route (including destination)",
    )
    intersections_remaining: int = Field(
        0, description="Number of intersections left before reaching the hospital"
    )

    # --- Hospital options ---
    hospitals: List[HospitalInfo] = Field(
        default_factory=list,
        description="All hospitals on the map with distance and specialization info",
    )
    target_hospital_id: Optional[str] = Field(
        None, description="Hospital selected by the agent (set after dispatch)"
    )

    # --- Lookahead window (next 3 intersections on route) ---
    lookahead_signals: List[SignalInfo] = Field(
        default_factory=list,
        description="Signal state for the next 1-3 intersections the ambulance will cross",
    )

    # --- Timing ---
    time_elapsed_seconds: float = Field(0.0, description="Simulated seconds since dispatch")
    time_limit_seconds: float = Field(
        300.0, description="Episode time limit in simulated seconds"
    )

    # --- Performance indicators ---
    last_speed_factor: float = Field(
        1.0,
        description="Ambulance speed last step: 1.0 = full speed, ~0.15 = blocked by red light",
    )
    stops_at_red: int = Field(
        0, description="Cumulative count of steps where ambulance was slowed by a red signal"
    )
    total_distance_covered: float = Field(
        0.0, description="Number of intersections the ambulance has passed so far"
    )

    # --- Signal efficiency metrics ---
    necessary_toggles: int = Field(
        0,
        description=(
            "Signals the agent correctly changed: wrong phase → right phase. "
            "These directly helped the ambulance."
        ),
    )
    unnecessary_toggles: int = Field(
        0,
        description=(
            "Signals the agent toggled that were already in the correct phase. "
            "Wasted actions — the ambulance would have passed green anyway."
        ),
    )
    first_signal_failures: int = Field(
        0,
        description=(
            "Times the ambulance hit a red at the immediately next intersection (S1). "
            "The agent had it in the lookahead window and still failed to clear it."
        ),
    )
    signal_efficiency: float = Field(
        1.0,
        description=(
            "necessary_toggles / max(1, necessary_toggles + unnecessary_toggles). "
            "1.0 = only acted on signals that needed changing. "
            "0.0 = all toggles were redundant."
        ),
    )


class AmbulanceState(State):
    """Episode-level metadata and cumulative statistics."""

    difficulty: str = Field("easy", description="'easy' | 'medium' | 'hard'")
    patient_condition: str = Field("cardiac")
    target_hospital_id: Optional[str] = Field(None)
    hospital_matched_condition: bool = Field(
        False,
        description="True if chosen hospital specialization matches patient condition",
    )
    total_stops: int = Field(0, description="Total red-light stops in this episode")
    arrival_time: Optional[float] = Field(
        None, description="Simulated seconds to reach hospital (None if not yet arrived)"
    )
    success: bool = Field(False, description="True if ambulance reached hospital in time")
