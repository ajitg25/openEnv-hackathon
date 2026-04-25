# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Ambulance Green Corridor Environment.

An LLM agent plays the role of both emergency dispatcher and city traffic
controller.  It must:
  1. (dispatch phase) Choose the most appropriate hospital for a patient, weighing
     distance, specialization match, and capacity.
  2. (routing phase) Control traffic signals in a rolling lookahead window so the
     ambulance always finds a green light ahead of it, minimising travel time.

City model
----------
- Grid of n×n intersections.
- Each intersection has a traffic signal: 'ns_green' (N↑/S↓ green, E/W red)
  or 'ew_green' (E→/W← green, N/S red).
- Road congestion is modelled as a density value 0–1 on every segment.
- The ambulance follows the BFS-shortest route from the patient to the chosen
  hospital.  Its speed at each step depends on the next intersection's signal
  phase and road density.

Difficulty levels
-----------------
  easy   – 6×6 grid, 2 general hospitals, cardiac patients only, low density
  medium – 8×8 grid, 3 specialised hospitals, mixed conditions, moderate density
  hard   – 12×12 grid, 5 hospitals (one at capacity), all conditions, high density
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Tuple
from uuid import uuid4

# Dual-import: relative in-repo, absolute in Docker.
try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
    from ..models import (
        AmbulanceAction,
        AmbulanceObservation,
        AmbulanceState,
        HospitalInfo,
        SignalControl,
        SignalInfo,
    )
except ImportError:
    from models import (  # type: ignore
        AmbulanceAction,
        AmbulanceObservation,
        AmbulanceState,
        HospitalInfo,
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

TIME_PER_STEP: float = 5.0          # simulated seconds per env step
SPEED_FULL: float = 0.8             # intersections/step when signal is green + clear road
SPEED_CONGESTED: float = 0.45       # green but dense traffic
SPEED_RED: float = 0.12             # signal is red (ambulance crawls)
LOOKAHEAD: int = 3                  # intersections ahead shown to agent
DENSITY_THRESHOLD: float = 0.45    # above this → congested speed


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
        "time_limit": 200.0,
        "base_density": 0.1,
        "signal_cycle_steps": 6,
    },
    "medium": {
        "grid_size": 8,
        "hospitals": [
            {"id": "hosp_a", "name": "City General",    "location": (0, 7), "specialization": "general"},
            {"id": "hosp_b", "name": "Cardiac Centre",  "location": (0, 3), "specialization": "cardiac"},
            {"id": "hosp_c", "name": "Trauma Centre",   "location": (2, 7), "specialization": "trauma"},
        ],
        "patient_pool": [(7, 0), (7, 4), (5, 2), (6, 6), (7, 7)],
        "conditions": ["cardiac", "trauma", "stroke"],
        "time_limit": 300.0,
        "base_density": 0.3,
        "signal_cycle_steps": 5,
    },
    "hard": {
        "grid_size": 12,
        "hospitals": [
            {"id": "hosp_a", "name": "City General",  "location": (0, 11), "specialization": "general"},
            {"id": "hosp_b", "name": "Cardiac Centre", "location": (0, 5),  "specialization": "cardiac"},
            {"id": "hosp_c", "name": "Trauma Centre",  "location": (2, 11), "specialization": "trauma"},
            {"id": "hosp_d", "name": "Stroke Centre",  "location": (0, 0),  "specialization": "stroke"},
            {"id": "hosp_e", "name": "East General",   "location": (5, 11), "specialization": "general",
             "at_capacity": True},
        ],
        "patient_pool": [(11, 0), (11, 6), (8, 3), (10, 9), (7, 5), (9, 1)],
        "conditions": ["cardiac", "trauma", "stroke"],
        "time_limit": 400.0,
        "base_density": 0.5,
        "signal_cycle_steps": 4,
    },
}

# Reward for hospital × condition match (base value multiplied in arrival reward)
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
# Helper
# ---------------------------------------------------------------------------

def _seg_key(a: Tuple[int, int], b: Tuple[int, int]) -> FrozenSet:
    return frozenset([a, b])


def _bfs_shortest(
    start: Tuple[int, int],
    end: Tuple[int, int],
    n: int,
) -> List[Tuple[int, int]]:
    """BFS shortest path on an n×n grid."""
    if start == end:
        return [start]
    visited = {start}
    queue: deque = deque([(start, [start])])
    while queue:
        (r, c), path = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in visited:
                new_path = path + [(nr, nc)]
                if (nr, nc) == end:
                    return new_path
                visited.add((nr, nc))
                queue.append(((nr, nc), new_path))
    return [start, end]


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


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class AmbulanceEnvironment(Environment):
    """
    Ambulance Green Corridor Routing Environment.

    Episode flow
    ------------
    1. reset() → observation with phase='dispatch' and list of hospitals.
    2. Agent calls step(AmbulanceAction(hospital_id='hosp_b')) → phase switches
       to 'routing', BFS route is computed.
    3. Agent calls step(AmbulanceAction(signal_controls=[...])) each tick,
       overriding up to LOOKAHEAD signals in the window ahead of the ambulance.
    4. Episode ends when the ambulance reaches the hospital or time runs out.

    Args:
        difficulty: 'easy' | 'medium' | 'hard'
        seed: Optional random seed for reproducibility.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, difficulty: str = "easy", seed: Optional[int] = None):
        self.difficulty = difficulty
        self._seed = seed
        self._cfg = DIFFICULTY_CONFIG[difficulty]
        self._n: int = self._cfg["grid_size"]

        # Episode variables – initialised properly in reset()
        self._patient_loc: Tuple[int, int] = (0, 0)
        self._patient_cond: str = "cardiac"
        self._hospitals: List[dict] = []
        self._signals: Dict[Tuple[int, int], str] = {}
        self._signal_steps: Dict[Tuple[int, int], int] = {}  # step counter per signal
        self._density: Dict[FrozenSet, float] = {}
        self._route: List[Tuple[int, int]] = []
        self._route_idx: int = 0
        self._progress: float = 0.0
        self._ambulance_loc: Tuple[int, int] = (0, 0)
        self._target_hospital_id: Optional[str] = None
        self._phase: str = "dispatch"
        self._time_elapsed: float = 0.0
        self._stops_at_red: int = 0
        self._speed_sum: float = 0.0
        self._routing_steps: int = 0
        # Signal efficiency counters
        self._necessary_toggles: int = 0
        self._unnecessary_toggles: int = 0
        self._first_signal_failures: int = 0
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
        if self._phase == "dispatch":
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
        self._phase = "dispatch"
        self._target_hospital_id = None
        self._route = []
        self._route_idx = 0
        self._progress = 0.0
        self._ambulance_loc = self._patient_loc
        self._time_elapsed = 0.0
        self._stops_at_red = 0
        self._speed_sum = 0.0
        self._routing_steps = 0
        self._necessary_toggles = 0
        self._unnecessary_toggles = 0
        self._first_signal_failures = 0
        self._init_signals()
        self._init_density()

    def _init_signals(self) -> None:
        """Checkerboard default: (r+c) even → ns_green, odd → ew_green."""
        for r in range(self._n):
            for c in range(self._n):
                self._signals[(r, c)] = "ns_green" if (r + c) % 2 == 0 else "ew_green"
                self._signal_steps[(r, c)] = 0

    def _init_density(self) -> None:
        base = self._cfg["base_density"]
        for r in range(self._n):
            for c in range(self._n):
                if c + 1 < self._n:
                    seg = _seg_key((r, c), (r, c + 1))
                    self._density[seg] = max(0.0, min(1.0, base + random.gauss(0, 0.08)))
                if r + 1 < self._n:
                    seg = _seg_key((r, c), (r + 1, c))
                    self._density[seg] = max(0.0, min(1.0, base + random.gauss(0, 0.08)))

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _dispatch_step(self, action: AmbulanceAction) -> Tuple[float, bool]:
        """Agent chooses a hospital."""
        if not action.hospital_id:
            return -10.0, False  # No choice yet – nudge agent

        chosen = next((h for h in self._hospitals if h["id"] == action.hospital_id), None)
        if chosen is None:
            return -10.0, False  # Unknown hospital ID

        if chosen.get("at_capacity", False):
            # Redirect to nearest available hospital
            available = [h for h in self._hospitals if not h.get("at_capacity", False)]
            if not available:
                return -200.0, True
            chosen = min(
                available,
                key=lambda h: abs(h["location"][0] - self._patient_loc[0])
                + abs(h["location"][1] - self._patient_loc[1]),
            )

        self._target_hospital_id = chosen["id"]
        self._episode_state.target_hospital_id = chosen["id"]

        # Plan route
        hospital_loc: Tuple[int, int] = tuple(chosen["location"])  # type: ignore[arg-type]
        self._route = _bfs_shortest(self._patient_loc, hospital_loc, self._n)
        self._route_idx = 0
        self._progress = 0.0
        self._ambulance_loc = self._patient_loc
        self._phase = "routing"

        # Up-front specialisation signal (10 % of final value)
        spec = SPEC_REWARD.get((self._patient_cond, chosen["specialization"]), 50.0)
        return spec * 0.1, False

    def _routing_step(self, action: AmbulanceAction) -> Tuple[float, bool]:
        """Agent controls signals; ambulance advances."""
        lookahead_infos = self._get_lookahead_infos()
        valid_positions = {(s.row, s.col): s for s in lookahead_infos}

        # 1. Classify and apply signal overrides
        toggle_penalty = 0.0
        for ctrl in action.signal_controls:
            pos = (ctrl.row, ctrl.col)
            if pos not in valid_positions or ctrl.phase not in ("ns_green", "ew_green"):
                continue
            sig = valid_positions[pos]
            needed_phase = (
                "ns_green" if sig.ambulance_direction in ("north", "south") else "ew_green"
            )
            already_correct = self._signals[pos] == needed_phase

            if already_correct:
                # Signal was already green for ambulance — toggling it is wasted effort
                self._unnecessary_toggles += 1
                toggle_penalty -= 2.0
            else:
                # Genuinely useful: signal was wrong, agent fixed it
                self._necessary_toggles += 1
                self._signals[pos] = ctrl.phase

        # 2. Advance ambient signal cycling (outside agent control)
        self._tick_signals(exclude=set(valid_positions.keys()))

        # 3. Check if the immediately next intersection (S1) is clear BEFORE advancing
        #    — captures "agent had it in view but still failed to clear it"
        if self._route_idx < len(self._route) - 1:
            next_pos = self._route[self._route_idx + 1]
            direction = _direction(self._route[self._route_idx], next_pos)
            if not _phase_allows(self._signals[next_pos], direction):
                self._first_signal_failures += 1

        # 4. Advance the ambulance
        speed = self._compute_speed()
        is_blocked = speed <= SPEED_RED + 0.01
        if is_blocked:
            self._stops_at_red += 1

        self._progress += speed
        self._speed_sum += speed
        self._routing_steps += 1

        while self._progress >= 1.0 and self._route_idx < len(self._route) - 1:
            self._progress -= 1.0
            self._route_idx += 1
            self._ambulance_loc = self._route[self._route_idx]

        # 5. Advance time
        self._time_elapsed += TIME_PER_STEP

        # 6. Check terminal conditions
        at_hospital = self._route_idx >= len(self._route) - 1
        timed_out = self._time_elapsed >= self._cfg["time_limit"]

        if at_hospital:
            return self._arrival_reward(), True
        if timed_out:
            return -500.0, True

        # Per-step reward: time penalty + speed bonus + toggle efficiency signal
        step_reward = -TIME_PER_STEP + 6.0 * speed + toggle_penalty
        return step_reward, False

    # ------------------------------------------------------------------
    # Signal cycling
    # ------------------------------------------------------------------

    def _tick_signals(self, exclude: set) -> None:
        """Advance natural signal cycle for intersections outside agent control."""
        cycle = self._cfg["signal_cycle_steps"]
        for pos in self._signals:
            if pos in exclude:
                continue
            self._signal_steps[pos] = (self._signal_steps[pos] + 1) % (cycle * 2)
            # Flip phase at the mid-point of the cycle
            if self._signal_steps[pos] == cycle:
                current = self._signals[pos]
                self._signals[pos] = "ew_green" if current == "ns_green" else "ns_green"

    # ------------------------------------------------------------------
    # Ambulance speed
    # ------------------------------------------------------------------

    def _compute_speed(self) -> float:
        """Speed toward the next route intersection, given current signal state."""
        if self._route_idx >= len(self._route) - 1:
            return SPEED_FULL  # Already at destination

        current = self._route[self._route_idx]
        nxt = self._route[self._route_idx + 1]
        direction = _direction(current, nxt)
        phase = self._signals[nxt]
        density = self._density.get(_seg_key(current, nxt), 0.0)

        if not _phase_allows(phase, direction):
            return SPEED_RED

        return SPEED_CONGESTED if density >= DENSITY_THRESHOLD else SPEED_FULL

    # ------------------------------------------------------------------
    # Arrival reward
    # ------------------------------------------------------------------

    def _arrival_reward(self) -> float:
        """Composite reward on hospital arrival."""
        base = 1000.0
        elapsed_fraction = self._time_elapsed / self._cfg["time_limit"]
        time_bonus = 500.0 * max(0.0, 1.0 - elapsed_fraction)

        chosen = next((h for h in self._hospitals if h["id"] == self._target_hospital_id), None)
        spec = 0.0
        matched = False
        if chosen:
            spec = SPEC_REWARD.get((self._patient_cond, chosen["specialization"]), 50.0)
            matched = chosen["specialization"] in (self._patient_cond, "general")

        stop_penalty = 20.0 * self._stops_at_red
        # Penalise wasteful toggling at episode end too
        waste_penalty = 5.0 * self._unnecessary_toggles

        self._episode_state.arrival_time = self._time_elapsed
        self._episode_state.success = True
        self._episode_state.total_stops = self._stops_at_red
        self._episode_state.hospital_matched_condition = matched
        self._episode_state.necessary_toggles = self._necessary_toggles
        self._episode_state.unnecessary_toggles = self._unnecessary_toggles
        self._episode_state.first_signal_failures = self._first_signal_failures
        total = self._necessary_toggles + self._unnecessary_toggles
        self._episode_state.signal_efficiency = (
            self._necessary_toggles / total if total > 0 else 1.0
        )

        return base + time_bonus + spec - stop_penalty - waste_penalty

    # ------------------------------------------------------------------
    # Lookahead helper
    # ------------------------------------------------------------------

    def _get_lookahead_infos(self) -> List[SignalInfo]:
        """Signal state for up to LOOKAHEAD intersections ahead on the route."""
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
            steps_since_last_tick = self._signal_steps[to_pos] % cycle
            seconds_until_change = (cycle - steps_since_last_tick) * TIME_PER_STEP
            density = self._density.get(_seg_key(from_pos, to_pos), 0.0)
            infos.append(
                SignalInfo(
                    row=to_pos[0],
                    col=to_pos[1],
                    phase=phase,
                    seconds_until_change=seconds_until_change,
                    traffic_density=round(density, 3),
                    ambulance_direction=direction,
                )
            )
        return infos

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _hospital_infos(self) -> List[HospitalInfo]:
        infos = []
        for h in self._hospitals:
            loc: Tuple[int, int] = tuple(h["location"])  # type: ignore[arg-type]
            dist = abs(loc[0] - self._patient_loc[0]) + abs(loc[1] - self._patient_loc[1])
            est_time = dist * TIME_PER_STEP / 0.7  # heuristic
            infos.append(
                HospitalInfo(
                    hospital_id=h["id"],
                    name=h["name"],
                    location=loc,
                    specialization=h["specialization"],
                    at_capacity=h.get("at_capacity", False),
                    distance_to_patient=dist,
                    travel_time_estimate=round(est_time, 1),
                )
            )
        return infos

    def _build_obs(self, reward: float, done: bool) -> AmbulanceObservation:
        avg_speed = self._speed_sum / max(1, self._routing_steps)
        remaining_route = list(self._route[self._route_idx:]) if self._route else []
        lookahead = self._get_lookahead_infos() if self._phase == "routing" else []

        total_toggles = self._necessary_toggles + self._unnecessary_toggles
        efficiency = self._necessary_toggles / total_toggles if total_toggles > 0 else 1.0

        return AmbulanceObservation(
            patient_location=self._patient_loc,
            patient_condition=self._patient_cond,
            phase=self._phase,
            ambulance_location=self._ambulance_loc,
            route_to_hospital=remaining_route,
            intersections_remaining=max(0, len(self._route) - self._route_idx - 1),
            hospitals=self._hospital_infos(),
            target_hospital_id=self._target_hospital_id,
            lookahead_signals=lookahead,
            time_elapsed_seconds=round(self._time_elapsed, 1),
            time_limit_seconds=self._cfg["time_limit"],
            last_speed_factor=round(avg_speed, 3),
            stops_at_red=self._stops_at_red,
            total_distance_covered=float(self._route_idx),
            necessary_toggles=self._necessary_toggles,
            unnecessary_toggles=self._unnecessary_toggles,
            first_signal_failures=self._first_signal_failures,
            signal_efficiency=round(efficiency, 3),
            done=done,
            reward=round(reward, 2),
        )
