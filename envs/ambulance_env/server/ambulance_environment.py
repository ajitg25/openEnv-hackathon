# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Ambulance Green Corridor Environment — LLM-Required Edition.

Why an LLM agent is genuinely necessary
----------------------------------------
- Traffic volume slows the ambulance EVEN on green signals.
- Road quality (potholes on 'damaged' roads) degrades speed independently.
- Dynamic mid-episode events (accidents, spikes, closures) block routes.
- The agent can switch hospitals or hint preferred directions at any step.
- Two alternative routes with ETAs are shown so the agent can reason about
  "slow short route vs fast long route".

Difficulty levels
-----------------
  easy   – 6×6 grid,  2 hospitals, 5 % event chance/step, ~20% damaged roads
  medium – 8×8 grid,  3 hospitals, 10% event chance/step, ~20% damaged roads
  hard   – 12×12 grid, 5 hospitals, 15% event chance/step, ~40% damaged roads
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
    from ..models import (
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
except ImportError:
    from models import (  # type: ignore
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
    try:
        from openenv.core.env_server.interfaces import Environment
        from openenv.core.env_server.types import State
    except ImportError:
        from openenv_core.env_server.interfaces import Environment  # type: ignore
        from openenv_core.env_server.types import State  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIME_PER_INTERSECTION: float = 10.0   # base seconds to cross one intersection
LOOKAHEAD: int = 3
HEAVY_TRAFFIC_THRESHOLD: float = 0.7

ROAD_TYPE_QUALITY: Dict[str, float] = {
    "highway": 1.0,
    "main": 0.75,
    "residential": 0.55,
    "damaged": 0.25,
}

# How signal state translates to a signal factor
SIGNAL_FACTOR_GREEN: float = 1.0
SIGNAL_FACTOR_RED: float = 0.15


# ---------------------------------------------------------------------------
# Difficulty configuration
# ---------------------------------------------------------------------------

DIFFICULTY_CONFIG: Dict[str, dict] = {
    "easy": {
        "grid_size": 6,
        "hospitals": [
            {"id": "hosp_a", "name": "City General",  "location": (0, 5), "specialization": "general"},
            {"id": "hosp_b", "name": "North General", "location": (0, 0), "specialization": "general"},
        ],
        "patient_pool": [(5, 0), (5, 3), (3, 2), (4, 4)],
        "conditions": ["cardiac"],
        "time_limit": 300.0,
        "base_traffic": 0.15,
        "signal_cycle_steps": 6,
        "event_prob": 0.05,
        "road_weights": {"highway": 0.2, "main": 0.6, "residential": 0.2, "damaged": 0.0},
    },
    "medium": {
        "grid_size": 8,
        "hospitals": [
            {"id": "hosp_a", "name": "City General",   "location": (0, 7), "specialization": "general"},
            {"id": "hosp_b", "name": "Cardiac Centre", "location": (0, 3), "specialization": "cardiac"},
            {"id": "hosp_c", "name": "Trauma Centre",  "location": (2, 7), "specialization": "trauma"},
        ],
        "patient_pool": [(7, 0), (7, 4), (5, 2), (6, 6), (7, 7)],
        "conditions": ["cardiac", "trauma", "stroke"],
        "time_limit": 400.0,
        "base_traffic": 0.35,
        "signal_cycle_steps": 5,
        "event_prob": 0.10,
        "road_weights": {"highway": 0.1, "main": 0.5, "residential": 0.3, "damaged": 0.1},
    },
    "hard": {
        "grid_size": 12,
        "hospitals": [
            {"id": "hosp_a", "name": "City General",   "location": (0, 11), "specialization": "general"},
            {"id": "hosp_b", "name": "Cardiac Centre", "location": (0, 5),  "specialization": "cardiac"},
            {"id": "hosp_c", "name": "Trauma Centre",  "location": (2, 11), "specialization": "trauma"},
            {"id": "hosp_d", "name": "Stroke Centre",  "location": (0, 0),  "specialization": "stroke"},
            {"id": "hosp_e", "name": "East General",   "location": (5, 11), "specialization": "general",
             "at_capacity": True},
        ],
        "patient_pool": [(11, 0), (11, 6), (8, 3), (10, 9), (7, 5), (9, 1)],
        "conditions": ["cardiac", "trauma", "stroke"],
        "time_limit": 500.0,
        "base_traffic": 0.55,
        "signal_cycle_steps": 4,
        "event_prob": 0.15,
        "road_weights": {"highway": 0.05, "main": 0.3, "residential": 0.25, "damaged": 0.4},
    },
}

SPEC_REWARD: Dict[Tuple[str, str], float] = {
    ("cardiac", "cardiac"): 300.0,
    ("cardiac", "general"):  100.0,
    ("cardiac", "trauma"):   -50.0,
    ("cardiac", "stroke"):   -50.0,
    ("trauma",  "trauma"):   300.0,
    ("trauma",  "general"):  100.0,
    ("trauma",  "cardiac"):  -50.0,
    ("trauma",  "stroke"):   -50.0,
    ("stroke",  "stroke"):   300.0,
    ("stroke",  "general"):  100.0,
    ("stroke",  "cardiac"):  -50.0,
    ("stroke",  "trauma"):   -50.0,
}


# ---------------------------------------------------------------------------
# Internal road segment data structure
# ---------------------------------------------------------------------------

@dataclass
class _Segment:
    quality: float
    road_type: str
    base_traffic: float
    current_traffic: float
    blocked: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg_key(a: Tuple[int, int], b: Tuple[int, int]) -> FrozenSet:
    return frozenset([a, b])


def _direction(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
    dr = to_pos[0] - from_pos[0]
    dc = to_pos[1] - from_pos[1]
    if dr < 0:
        return "north"
    if dr > 0:
        return "south"
    if dc < 0:
        return "west"
    return "east"


def _phase_allows(phase: str, direction: str) -> bool:
    if phase == "ns_green":
        return direction in ("north", "south")
    return direction in ("east", "west")


def _segment_speed(seg: _Segment, signal_factor: float) -> float:
    """Compute 0.0-1.0 speed factor using traffic + quality + signal formula."""
    traffic_factor = max(0.2, 1.0 - 0.75 * seg.current_traffic)
    quality_factor = max(0.3, seg.quality)
    return signal_factor * traffic_factor * quality_factor


def _choose_road_type(weights: Dict[str, float]) -> str:
    types = list(weights.keys())
    probs = [weights[t] for t in types]
    return random.choices(types, weights=probs, k=1)[0]


# ---------------------------------------------------------------------------
# A* routing
# ---------------------------------------------------------------------------

def _astar(
    start: Tuple[int, int],
    end: Tuple[int, int],
    n: int,
    segments: Dict[FrozenSet, _Segment],
    signals: Dict[Tuple[int, int], str],
) -> List[Tuple[int, int]]:
    """A* shortest path weighted by estimated transit time. Skips blocked segments."""
    if start == end:
        return [start]

    def heuristic(pos: Tuple[int, int]) -> float:
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    open_heap: list = []
    heapq.heappush(open_heap, (heuristic(start), 0.0, start, [start]))
    visited: Dict[Tuple[int, int], float] = {}

    while open_heap:
        _, cost, pos, path = heapq.heappop(open_heap)

        if pos == end:
            return path
        if pos in visited and visited[pos] <= cost:
            continue
        visited[pos] = cost

        r, c = pos
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            neighbor = (nr, nc)
            key = _seg_key(pos, neighbor)
            seg = segments.get(key)
            if seg is None or seg.blocked:
                continue
            direction = _direction(pos, neighbor)
            phase = signals.get(neighbor, "ns_green")
            signal_factor = SIGNAL_FACTOR_GREEN if _phase_allows(phase, direction) else SIGNAL_FACTOR_RED
            speed = _segment_speed(seg, signal_factor)
            # Avoid division by zero; very slow segments have large cost
            transit_time = TIME_PER_INTERSECTION / max(0.01, speed)
            new_cost = cost + transit_time
            f = new_cost + heuristic(neighbor) * TIME_PER_INTERSECTION
            heapq.heappush(open_heap, (f, new_cost, neighbor, path + [neighbor]))

    # Fallback: straight-line if no path found (shouldn't happen on full grid)
    return [start, end]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class AmbulanceEnvironment(Environment):
    """
    Ambulance Green Corridor Routing Environment.

    Episode flow
    ------------
    1. reset() → initial observation (no route yet; agent must set hospital_id).
    2. Agent calls step(AmbulanceAction(hospital_id='hosp_b')) → route is computed.
    3. Agent calls step() each tick, controlling signals and optionally switching
       hospital or hinting a preferred direction.
    4. Episode ends when ambulance reaches hospital or time runs out.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, difficulty: str = "easy", seed: Optional[int] = None):
        self.difficulty = difficulty
        self._seed = seed
        self._cfg = DIFFICULTY_CONFIG[difficulty]
        self._n: int = self._cfg["grid_size"]

        # Initialised in reset()
        self._patient_loc: Tuple[int, int] = (0, 0)
        self._patient_cond: str = "cardiac"
        self._hospitals: List[dict] = []
        self._signals: Dict[Tuple[int, int], str] = {}
        self._signal_steps: Dict[Tuple[int, int], int] = {}
        self._segments: Dict[FrozenSet, _Segment] = {}
        self._route: List[Tuple[int, int]] = []
        self._route_idx: int = 0
        self._progress: float = 0.0
        self._ambulance_loc: Tuple[int, int] = (0, 0)
        self._target_hospital_id: Optional[str] = None
        self._target_hospital_loc: Optional[Tuple[int, int]] = None
        self._time_elapsed: float = 0.0
        self._last_speed_factor: float = 1.0
        self._stops_at_red: int = 0
        self._necessary_toggles: int = 0
        self._unnecessary_toggles: int = 0
        self._first_signal_failures: int = 0
        self._successful_reroutes: int = 0
        self._damaged_segments_traversed: int = 0
        self._active_events: List[DynamicEvent] = []   # new events this step only
        self._event_affected_segments: set = set()      # segments already hit by an event
        self._dispatched: bool = False
        self._episode_state: AmbulanceState = AmbulanceState(
            episode_id=str(uuid4()), step_count=0, difficulty=difficulty
        )

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> AmbulanceObservation:
        if seed is not None:
            random.seed(seed)
        elif self._seed is not None:
            random.seed(self._seed)

        self._init_episode()
        self._episode_state = AmbulanceState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            difficulty=self.difficulty,
            patient_condition=self._patient_cond,
        )
        return self._build_obs(reward=0.0, done=False)

    def step(self, action: AmbulanceAction, **kwargs) -> AmbulanceObservation:  # type: ignore[override]
        self._episode_state.step_count += 1

        if not self._dispatched:
            reward, done = self._dispatch_step(action)
        else:
            reward, done = self._routing_step(action)

        return self._build_obs(reward=reward, done=done)

    @property
    def state(self) -> AmbulanceState:  # type: ignore[override]
        return self._episode_state

    # ------------------------------------------------------------------
    # Episode initialisation
    # ------------------------------------------------------------------

    def _init_episode(self) -> None:
        cfg = self._cfg
        self._patient_loc = random.choice(cfg["patient_pool"])
        self._patient_cond = random.choice(cfg["conditions"])
        self._hospitals = [dict(h) for h in cfg["hospitals"]]
        self._dispatched = False
        self._target_hospital_id = None
        self._target_hospital_loc = None
        self._route = []
        self._route_idx = 0
        self._progress = 0.0
        self._ambulance_loc = self._patient_loc
        self._time_elapsed = 0.0
        self._last_speed_factor = 1.0
        self._stops_at_red = 0
        self._necessary_toggles = 0
        self._unnecessary_toggles = 0
        self._first_signal_failures = 0
        self._successful_reroutes = 0
        self._damaged_segments_traversed = 0
        self._active_events = []
        self._event_affected_segments = set()
        self._init_signals()
        self._init_road_network()

    def _init_signals(self) -> None:
        for r in range(self._n):
            for c in range(self._n):
                self._signals[(r, c)] = "ns_green" if (r + c) % 2 == 0 else "ew_green"
                self._signal_steps[(r, c)] = 0

    def _init_road_network(self) -> None:
        """Build segment graph with randomised road types and traffic."""
        weights = self._cfg["road_weights"]
        base_traffic = self._cfg["base_traffic"]
        self._segments = {}

        for r in range(self._n):
            for c in range(self._n):
                for dr, dc in ((0, 1), (1, 0)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < self._n and 0 <= nc < self._n):
                        continue
                    road_type = _choose_road_type(weights)
                    quality = ROAD_TYPE_QUALITY[road_type]
                    traffic = max(0.0, min(1.0, base_traffic + random.gauss(0, 0.1)))
                    key = _seg_key((r, c), (nr, nc))
                    self._segments[key] = _Segment(
                        quality=quality,
                        road_type=road_type,
                        base_traffic=traffic,
                        current_traffic=traffic,
                    )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch_step(self, action: AmbulanceAction) -> Tuple[float, bool]:
        hospital_id = action.hospital_id
        if not hospital_id:
            return -10.0, False

        chosen = next((h for h in self._hospitals if h["id"] == hospital_id), None)
        if chosen is None:
            return -10.0, False

        if chosen.get("at_capacity", False):
            available = [h for h in self._hospitals if not h.get("at_capacity", False)]
            if not available:
                return -200.0, True
            chosen = min(
                available,
                key=lambda h: abs(h["location"][0] - self._patient_loc[0])
                + abs(h["location"][1] - self._patient_loc[1]),
            )

        self._set_target_hospital(chosen)
        self._dispatched = True
        spec = SPEC_REWARD.get((self._patient_cond, chosen["specialization"]), 50.0)
        return spec * 0.1, False

    def _set_target_hospital(self, hospital: dict) -> None:
        self._target_hospital_id = hospital["id"]
        self._target_hospital_loc = tuple(hospital["location"])  # type: ignore[assignment]
        self._episode_state.target_hospital_id = hospital["id"]
        self._recompute_route()

    def _recompute_route(self) -> None:
        if self._target_hospital_loc is None:
            return
        self._route = _astar(
            self._ambulance_loc,
            self._target_hospital_loc,
            self._n,
            self._segments,
            self._signals,
        )
        self._route_idx = 0
        self._progress = 0.0

    # ------------------------------------------------------------------
    # Routing step
    # ------------------------------------------------------------------

    def _routing_step(self, action: AmbulanceAction) -> Tuple[float, bool]:
        # 1. Optional: switch hospital destination
        if action.hospital_id and action.hospital_id != self._target_hospital_id:
            new_hosp = next((h for h in self._hospitals if h["id"] == action.hospital_id), None)
            if new_hosp and not new_hosp.get("at_capacity", False):
                self._set_target_hospital(new_hosp)
                self._successful_reroutes += 1

        # 2. Apply signal controls
        lookahead_infos = self._get_lookahead_infos()
        valid_positions = {(s.row, s.col): s for s in lookahead_infos}
        self._apply_signal_controls(action.signal_controls, valid_positions)

        # 3. Handle preferred_direction hint
        if action.preferred_direction and self._route:
            self._apply_preferred_direction(action.preferred_direction)

        # 4. Tick ambient signals
        self._tick_signals(exclude=set(valid_positions.keys()))

        # 5. Check if next intersection is already red (failure tracking)
        self._check_first_signal_failure()

        # 6. Compute speed and advance ambulance
        speed_factor = self._advance_ambulance()

        # 7. Fire dynamic events — reset each step so observation shows only new ones
        self._active_events = self._fire_dynamic_events()

        # 8. Check if current route is blocked → force re-route
        if self._route_blocked():
            self._recompute_route()
            if new_events:
                self._successful_reroutes += 1

        # 9. Advance time
        self._time_elapsed += TIME_PER_INTERSECTION

        # 10. Check terminal conditions
        at_hospital = self._route_idx >= len(self._route) - 1
        timed_out = self._time_elapsed >= self._cfg["time_limit"]

        if at_hospital:
            return self._arrival_reward(), True
        if timed_out:
            return -500.0, True

        waste_penalty = 5.0 * (self._unnecessary_toggles - getattr(self, "_prev_unnecessary", 0))
        self._prev_unnecessary = self._unnecessary_toggles
        step_reward = -TIME_PER_INTERSECTION + 20.0 * speed_factor + waste_penalty
        return step_reward, False

    def _apply_signal_controls(
        self,
        controls: List[SignalControl],
        valid_positions: dict,
    ) -> None:
        for ctrl in controls:
            pos = (ctrl.row, ctrl.col)
            if pos not in valid_positions or ctrl.phase not in ("ns_green", "ew_green"):
                continue
            sig = valid_positions[pos]
            needed_phase = (
                "ns_green" if sig.ambulance_direction in ("north", "south") else "ew_green"
            )
            if self._signals[pos] == needed_phase:
                self._unnecessary_toggles += 1
            else:
                self._necessary_toggles += 1
                self._signals[pos] = ctrl.phase

    def _apply_preferred_direction(self, preferred: str) -> None:
        """If there's a valid neighbor in the preferred direction, re-route through it."""
        cur = self._ambulance_loc
        delta = {"north": (-1, 0), "south": (1, 0), "west": (0, -1), "east": (0, 1)}
        if preferred not in delta:
            return
        dr, dc = delta[preferred]
        neighbor = (cur[0] + dr, cur[1] + dc)
        if not (0 <= neighbor[0] < self._n and 0 <= neighbor[1] < self._n):
            return
        key = _seg_key(cur, neighbor)
        seg = self._segments.get(key)
        if seg is None or seg.blocked or self._target_hospital_loc is None:
            return
        # Recompute route from this forced neighbor onward
        tail = _astar(neighbor, self._target_hospital_loc, self._n, self._segments, self._signals)
        self._route = [cur] + tail
        self._route_idx = 0
        self._progress = 0.0

    def _check_first_signal_failure(self) -> None:
        if self._route_idx < len(self._route) - 1:
            next_pos = self._route[self._route_idx + 1]
            direction = _direction(self._route[self._route_idx], next_pos)
            if not _phase_allows(self._signals[next_pos], direction):
                self._first_signal_failures += 1

    def _advance_ambulance(self) -> float:
        """Move ambulance along route; return speed factor used."""
        if self._route_idx >= len(self._route) - 1:
            self._last_speed_factor = 1.0
            return 1.0

        current = self._route[self._route_idx]
        nxt = self._route[self._route_idx + 1]
        key = _seg_key(current, nxt)
        seg = self._segments.get(key, _Segment(quality=0.75, road_type="main", base_traffic=0.2, current_traffic=0.2))

        direction = _direction(current, nxt)
        phase = self._signals.get(nxt, "ns_green")
        signal_factor = SIGNAL_FACTOR_GREEN if _phase_allows(phase, direction) else SIGNAL_FACTOR_RED
        speed = _segment_speed(seg, signal_factor)

        if signal_factor == SIGNAL_FACTOR_RED:
            self._stops_at_red += 1
        if seg.road_type == "damaged":
            self._damaged_segments_traversed += 1

        self._last_speed_factor = speed
        self._progress += speed

        while self._progress >= 1.0 and self._route_idx < len(self._route) - 1:
            self._progress -= 1.0
            self._route_idx += 1
            self._ambulance_loc = self._route[self._route_idx]

        return speed

    def _route_blocked(self) -> bool:
        """Return True if any upcoming segment on the current route is blocked."""
        for i in range(self._route_idx, min(self._route_idx + 5, len(self._route) - 1)):
            key = _seg_key(self._route[i], self._route[i + 1])
            seg = self._segments.get(key)
            if seg and seg.blocked:
                return True
        return False

    # ------------------------------------------------------------------
    # Dynamic events
    # ------------------------------------------------------------------

    def _fire_dynamic_events(self) -> List[DynamicEvent]:
        prob = self._cfg["event_prob"]
        if random.random() > prob:
            return []

        event_type = random.choice(["accident", "traffic_spike", "road_closure"])
        # Pick a random segment not on the immediate next 2 steps of route
        safe_positions = {
            _seg_key(self._route[i], self._route[i + 1])
            for i in range(self._route_idx, min(self._route_idx + 2, len(self._route) - 1))
        }
        candidates = [k for k in self._segments if k not in safe_positions and k not in self._event_affected_segments]
        if not candidates:
            return []

        key = random.choice(candidates)
        self._event_affected_segments.add(key)
        seg = self._segments[key]
        severity = round(random.uniform(0.5, 1.0), 2)
        pos_list = list(key)
        position = pos_list[0] if isinstance(pos_list[0], tuple) else tuple(pos_list[0])

        if event_type == "accident":
            seg.blocked = True
            desc = f"Accident at {position} blocking the road (severity={severity})"
        elif event_type == "traffic_spike":
            seg.current_traffic = min(1.0, seg.current_traffic + 0.4 + 0.5 * severity)
            desc = f"Traffic spike near {position}, volume={seg.current_traffic:.2f}"
        else:  # road_closure
            seg.blocked = True
            desc = f"Road closure at {position} due to construction (severity={severity})"

        return [DynamicEvent(
            event_type=event_type,
            position=position,  # type: ignore[arg-type]
            severity=severity,
            description=desc,
        )]

    # ------------------------------------------------------------------
    # Signal cycling
    # ------------------------------------------------------------------

    def _tick_signals(self, exclude: set) -> None:
        cycle = self._cfg["signal_cycle_steps"]
        for pos in self._signals:
            if pos in exclude:
                continue
            self._signal_steps[pos] = (self._signal_steps[pos] + 1) % (cycle * 2)
            if self._signal_steps[pos] == cycle:
                current = self._signals[pos]
                self._signals[pos] = "ew_green" if current == "ns_green" else "ns_green"

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _arrival_reward(self) -> float:
        arrival_bonus = 1000.0
        time_bonus = max(0.0, 500.0 * (1.0 - self._time_elapsed / self._cfg["time_limit"]))

        chosen = next((h for h in self._hospitals if h["id"] == self._target_hospital_id), None)
        spec = 0.0
        matched = False
        if chosen:
            spec = SPEC_REWARD.get((self._patient_cond, chosen["specialization"]), 50.0)
            matched = chosen["specialization"] in (self._patient_cond, "general")

        stop_penalty = 20.0 * self._stops_at_red
        waste_penalty = 5.0 * self._unnecessary_toggles
        comfort_penalty = 10.0 * self._damaged_segments_traversed
        reroute_bonus = 50.0 * self._successful_reroutes

        self._episode_state.arrival_time = self._time_elapsed
        self._episode_state.success = True
        self._episode_state.total_stops = self._stops_at_red
        self._episode_state.hospital_matched_condition = matched
        self._episode_state.necessary_toggles = self._necessary_toggles
        self._episode_state.unnecessary_toggles = self._unnecessary_toggles
        self._episode_state.first_signal_failures = self._first_signal_failures
        self._episode_state.successful_reroutes = self._successful_reroutes
        self._episode_state.damaged_segments_traversed = self._damaged_segments_traversed
        total = self._necessary_toggles + self._unnecessary_toggles
        self._episode_state.signal_efficiency = (
            self._necessary_toggles / total if total > 0 else 1.0
        )

        return arrival_bonus + time_bonus + spec - stop_penalty - waste_penalty - comfort_penalty + reroute_bonus

    # ------------------------------------------------------------------
    # Route building helpers
    # ------------------------------------------------------------------

    def _compute_route_option(
        self,
        hospital: dict,
        start: Optional[Tuple[int, int]] = None,
    ) -> RouteOption:
        origin = start or self._ambulance_loc
        hosp_loc: Tuple[int, int] = tuple(hospital["location"])  # type: ignore[assignment]
        path = _astar(origin, hosp_loc, self._n, self._segments, self._signals)
        segments = self._path_to_segments(path)
        est_time = self._estimated_time(path)
        num_damaged = sum(1 for s in segments if s.road_type == "damaged")
        num_heavy = sum(1 for s in segments if s.traffic_volume > HEAVY_TRAFFIC_THRESHOLD)
        return RouteOption(
            hospital_id=hospital["id"],
            hospital_name=hospital["name"],
            path=path,
            segments=segments,
            estimated_time=round(est_time, 1),
            num_damaged_segments=num_damaged,
            num_heavy_traffic_segments=num_heavy,
        )

    def _path_to_segments(self, path: List[Tuple[int, int]]) -> List[RoadSegment]:
        result = []
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            key = _seg_key(a, b)
            seg = self._segments.get(key)
            if seg is None:
                continue
            direction = _direction(a, b)
            phase = self._signals.get(b, "ns_green")
            sig_factor = SIGNAL_FACTOR_GREEN if _phase_allows(phase, direction) else SIGNAL_FACTOR_RED
            speed = _segment_speed(seg, sig_factor)
            transit = TIME_PER_INTERSECTION / max(0.01, speed)
            result.append(RoadSegment(
                from_pos=a,
                to_pos=b,
                direction=direction,
                road_type=seg.road_type,
                road_quality=round(seg.quality, 3),
                traffic_volume=round(seg.current_traffic, 3),
                blocked=seg.blocked,
                estimated_transit_time=round(transit, 1),
            ))
        return result

    def _estimated_time(self, path: List[Tuple[int, int]]) -> float:
        total = 0.0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            key = _seg_key(a, b)
            seg = self._segments.get(key)
            if seg is None:
                total += TIME_PER_INTERSECTION
                continue
            direction = _direction(a, b)
            phase = self._signals.get(b, "ns_green")
            sig_factor = SIGNAL_FACTOR_GREEN if _phase_allows(phase, direction) else SIGNAL_FACTOR_RED
            speed = _segment_speed(seg, sig_factor)
            total += TIME_PER_INTERSECTION / max(0.01, speed)
        return total

    def _get_alternative_routes(self) -> List[RouteOption]:
        """Return up to 2 route options: different hospitals or alternate paths."""
        alts = []
        current_hosp_id = self._target_hospital_id

        # First alternative: best non-target hospital
        for hosp in self._hospitals:
            if hosp["id"] == current_hosp_id:
                continue
            if hosp.get("at_capacity", False):
                continue
            alts.append(self._compute_route_option(hosp))
            if len(alts) >= 2:
                break

        # Sort by estimated time so agent sees the fastest alternative first
        alts.sort(key=lambda r: r.estimated_time)
        return alts[:2]

    # ------------------------------------------------------------------
    # Lookahead signals
    # ------------------------------------------------------------------

    def _get_lookahead_infos(self) -> List[SignalInfo]:
        infos: List[SignalInfo] = []
        if not self._route:
            return infos
        cycle = self._cfg["signal_cycle_steps"]
        for i in range(1, LOOKAHEAD + 1):
            from_idx = self._route_idx + i - 1
            to_idx = self._route_idx + i
            if to_idx >= len(self._route):
                break
            from_pos = self._route[from_idx]
            to_pos = self._route[to_idx]
            direction = _direction(from_pos, to_pos)
            phase = self._signals[to_pos]
            steps_since = self._signal_steps[to_pos] % cycle
            seconds_until_change = (cycle - steps_since) * TIME_PER_INTERSECTION
            key = _seg_key(from_pos, to_pos)
            seg = self._segments.get(key)
            density = seg.current_traffic if seg else 0.0
            infos.append(SignalInfo(
                row=to_pos[0],
                col=to_pos[1],
                phase=phase,
                seconds_until_change=seconds_until_change,
                traffic_density=round(density, 3),
                ambulance_direction=direction,
            ))
        return infos

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _hospital_infos(self) -> List[HospitalInfo]:
        infos = []
        for h in self._hospitals:
            loc: Tuple[int, int] = tuple(h["location"])  # type: ignore[assignment]
            dist = abs(loc[0] - self._patient_loc[0]) + abs(loc[1] - self._patient_loc[1])
            # Estimate based on A* if dispatched, otherwise Manhattan heuristic
            if self._dispatched:
                path = _astar(self._ambulance_loc, loc, self._n, self._segments, self._signals)
                est_time = self._estimated_time(path)
            else:
                est_time = dist * TIME_PER_INTERSECTION / 0.6
            infos.append(HospitalInfo(
                hospital_id=h["id"],
                name=h["name"],
                location=loc,
                specialization=h["specialization"],
                at_capacity=h.get("at_capacity", False),
                distance_to_patient=dist,
                travel_time_estimate=round(est_time, 1),
            ))
        return infos

    def _current_route_option(self) -> RouteOption:
        """Build RouteOption for the active route from current position onward."""
        if not self._route or self._target_hospital_id is None:
            # Return a dummy route before dispatch
            hosp = self._hospitals[0]
            return RouteOption(
                hospital_id=hosp["id"],
                hospital_name=hosp["name"],
                path=[self._ambulance_loc],
                segments=[],
                estimated_time=0.0,
                num_damaged_segments=0,
                num_heavy_traffic_segments=0,
            )
        remaining_path = self._route[self._route_idx:]
        segments = self._path_to_segments(remaining_path)
        est_time = self._estimated_time(remaining_path)
        num_damaged = sum(1 for s in segments if s.road_type == "damaged")
        num_heavy = sum(1 for s in segments if s.traffic_volume > HEAVY_TRAFFIC_THRESHOLD)
        hosp = next((h for h in self._hospitals if h["id"] == self._target_hospital_id), self._hospitals[0])
        return RouteOption(
            hospital_id=hosp["id"],
            hospital_name=hosp["name"],
            path=remaining_path,
            segments=segments,
            estimated_time=round(est_time, 1),
            num_damaged_segments=num_damaged,
            num_heavy_traffic_segments=num_heavy,
        )

    def _current_segment_info(self) -> Optional[RoadSegment]:
        if not self._route or self._route_idx >= len(self._route) - 1:
            return None
        a = self._route[self._route_idx]
        b = self._route[self._route_idx + 1]
        key = _seg_key(a, b)
        seg = self._segments.get(key)
        if seg is None:
            return None
        direction = _direction(a, b)
        phase = self._signals.get(b, "ns_green")
        sig_factor = SIGNAL_FACTOR_GREEN if _phase_allows(phase, direction) else SIGNAL_FACTOR_RED
        speed = _segment_speed(seg, sig_factor)
        transit = TIME_PER_INTERSECTION / max(0.01, speed)
        return RoadSegment(
            from_pos=a,
            to_pos=b,
            direction=direction,
            road_type=seg.road_type,
            road_quality=round(seg.quality, 3),
            traffic_volume=round(seg.current_traffic, 3),
            blocked=seg.blocked,
            estimated_transit_time=round(transit, 1),
        )

    def _build_obs(self, reward: float, done: bool) -> AmbulanceObservation:
        total_toggles = self._necessary_toggles + self._unnecessary_toggles
        efficiency = self._necessary_toggles / total_toggles if total_toggles > 0 else 1.0
        lookahead = self._get_lookahead_infos() if self._dispatched else []
        alt_routes = self._get_alternative_routes() if self._dispatched else []

        return AmbulanceObservation(
            patient_location=self._patient_loc,
            patient_condition=self._patient_cond,
            ambulance_location=self._ambulance_loc,
            current_segment=self._current_segment_info(),
            current_route=self._current_route_option(),
            alternative_routes=alt_routes,
            target_hospital_id=self._target_hospital_id,
            intersections_remaining=max(0, len(self._route) - self._route_idx - 1),
            lookahead_signals=lookahead,
            active_events=list(self._active_events),
            hospitals=self._hospital_infos(),
            time_elapsed_seconds=round(self._time_elapsed, 1),
            time_limit_seconds=self._cfg["time_limit"],
            last_speed_factor=round(self._last_speed_factor, 3),
            stops_at_red=self._stops_at_red,
            total_distance_covered=float(self._route_idx),
            necessary_toggles=self._necessary_toggles,
            unnecessary_toggles=self._unnecessary_toggles,
            first_signal_failures=self._first_signal_failures,
            signal_efficiency=round(efficiency, 3),
            successful_reroutes=self._successful_reroutes,
            damaged_segments_traversed=self._damaged_segments_traversed,
            done=done,
            reward=round(reward, 2),
        )
