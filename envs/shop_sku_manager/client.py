# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shop SKU Manager Environment Client.

Type-safe WebSocket client for training agents on inventory management tasks.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .models import OrderAction, ShopObservation, ShopState


class ShopSKUManagerEnv(EnvClient[OrderAction, ShopObservation, ShopState]):
    """
    WebSocket client for Shop SKU Manager environment.

    Example usage:
        ```python
        env = ShopSKUManagerEnv(base_url="http://localhost:8000")

        obs = await env.reset(task_difficulty="easy")

        while not obs.done:
            action = OrderAction(orders={"milk": 10, "bread": 5})
            obs = await env.step(action)

        state = await env.state()
        print(f"Final profit: ${state.total_profit:.2f}")
        ```
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize environment client.

        Args:
            base_url: URL of the environment server
        """
        super().__init__(base_url=base_url)

    def _step_payload(self, action: OrderAction) -> Dict[str, Any]:
        """Convert typed action to JSON for WebSocket message."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ShopObservation]:
        """Parse JSON response into typed observation."""
        obs_data = payload.get("observation", {})

        observation = ShopObservation(
            inventory_levels=obs_data.get("inventory_levels", {}),
            sales_last_7_days=obs_data.get("sales_last_7_days", {}),
            demand_forecast=obs_data.get("demand_forecast", {}),
            actual_demand=obs_data.get("actual_demand", {}),
            reorder_pending=obs_data.get("reorder_pending", {}),
            lead_times=obs_data.get("lead_times", {}),
            storage_cost_per_unit=obs_data.get("storage_cost_per_unit", {}),
            unit_cost=obs_data.get("unit_cost", {}),
            unit_price=obs_data.get("unit_price", {}),
            budget_remaining=obs_data.get("budget_remaining", 0.0),
            supplier_min_orders=obs_data.get("supplier_min_orders", {}),
            day_of_week=obs_data.get("day_of_week", 0),
            season=obs_data.get("season", "spring"),
            current_day=obs_data.get("current_day", 0),
            stockout_flags=obs_data.get("stockout_flags", {}),
            stockout_days=obs_data.get("stockout_days", {}),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ShopState:
        """Parse JSON response into state."""
        state_data = payload.get("state", {})

        return ShopState(
            episode_id=state_data.get("episode_id", ""),
            step_count=state_data.get("step_count", 0),
            task_difficulty=state_data.get("task_difficulty", "easy"),
            total_revenue=state_data.get("total_revenue", 0.0),
            total_cost=state_data.get("total_cost", 0.0),
            total_profit=state_data.get("total_profit", 0.0),
            total_reward=state_data.get("total_reward", 0.0),
            stockout_days_count=state_data.get("stockout_days_count", 0),
            excess_inventory=state_data.get("excess_inventory", 0.0),
            emergency_orders_count=state_data.get("emergency_orders_count", 0),
            num_skus=state_data.get("num_skus", 0),
            episode_length=state_data.get("episode_length", 30),
        )
