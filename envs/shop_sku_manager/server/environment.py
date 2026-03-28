# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shop SKU Manager Environment Implementation.

Simulates a retail shop inventory management task.
Agent learns when to order products and how to manage demand.
"""

from __future__ import annotations

import random
from uuid import uuid4
from collections import defaultdict
from typing import Dict, List

import numpy as np
from openenv.core.env_server.interfaces import Environment

from ..models import OrderAction, ShopObservation, ShopState


class ShopSKUManagerEnvironment(Environment):
    """
    Environment: Shop inventory management.

    Agent controls:
    - When to order products
    - How much to order
    - Whether to use emergency shipping

    Goal: Maximize profit = Revenue - Ordering Cost - Storage Cost - Stockout Penalties
    """

    # SKU definitions for different difficulties
    SKU_DATABASE = {
        "easy": {
            "milk": {"base_demand": 20, "variability": 0.1, "lead_time": 2},
            "bread": {"base_demand": 15, "variability": 0.1, "lead_time": 1},
            "eggs": {"base_demand": 10, "variability": 0.1, "lead_time": 2},
        },
        "medium": {
            "milk": {"base_demand": 20, "variability": 0.3, "lead_time": 2},
            "bread": {"base_demand": 15, "variability": 0.3, "lead_time": 1},
            "eggs": {"base_demand": 10, "variability": 0.3, "lead_time": 2},
            "cheese": {"base_demand": 8, "variability": 0.2, "lead_time": 3},
            "yogurt": {"base_demand": 12, "variability": 0.2, "lead_time": 2},
        },
        "hard": {
            "milk": {"base_demand": 20, "variability": 0.4, "lead_time": 2},
            "bread": {"base_demand": 15, "variability": 0.4, "lead_time": 1},
            "eggs": {"base_demand": 10, "variability": 0.4, "lead_time": 2},
            "cheese": {"base_demand": 8, "variability": 0.3, "lead_time": 3},
            "yogurt": {"base_demand": 12, "variability": 0.3, "lead_time": 2},
            "butter": {"base_demand": 5, "variability": 0.35, "lead_time": 3},
            "cream": {"base_demand": 4, "variability": 0.4, "lead_time": 3},
        },
    }

    UNIT_COSTS = {
        "milk": 2.0,
        "bread": 1.0,
        "eggs": 1.5,
        "cheese": 3.0,
        "yogurt": 1.2,
        "butter": 4.0,
        "cream": 3.5,
    }

    UNIT_PRICES = {
        "milk": 4.0,
        "bread": 2.5,
        "eggs": 3.0,
        "cheese": 6.0,
        "yogurt": 2.5,
        "butter": 8.0,
        "cream": 7.0,
    }

    STORAGE_COST_PER_UNIT = {
        "milk": 0.05,
        "bread": 0.02,
        "eggs": 0.03,
        "cheese": 0.08,
        "yogurt": 0.04,
        "butter": 0.1,
        "cream": 0.1,
    }

    def __init__(
        self,
        task_difficulty: str = "easy",
        episode_length: int = 30,
        initial_budget: float = 500.0,
    ):
        """
        Initialize environment.

        Args:
            task_difficulty: "easy", "medium", or "hard"
            episode_length: Number of days per episode
            initial_budget: Starting budget for orders
        """
        self.task_difficulty = task_difficulty
        self.episode_length = episode_length
        self.initial_budget = initial_budget

        # Get SKUs for this difficulty
        self.skus = list(self.SKU_DATABASE[task_difficulty].keys())
        self.sku_configs = self.SKU_DATABASE[task_difficulty]

        # Initialize state
        self._reset_episode()

    def _reset_episode(self):
        """Reset all episode state."""
        self.episode_id = str(uuid4())
        self.current_day = 0

        # Initialize inventory (start with some stock)
        self.inventory = {sku: 30.0 for sku in self.skus}

        # Orders in transit: {sku: {arrival_day: quantity}}
        self.orders_in_transit: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        # Tracking
        self.total_revenue = 0.0
        self.total_cost = 0.0
        self.total_reward = 0.0
        self.budget_remaining = self.initial_budget
        self.stockout_days = {sku: 0 for sku in self.skus}
        self.emergency_orders = 0

        # History for demand forecasting
        self.sales_history = {sku: [] for sku in self.skus}

        # Current state object
        self._state = ShopState(
            episode_id=self.episode_id,
            step_count=0,
            task_difficulty=self.task_difficulty,
            num_skus=len(self.skus),
            episode_length=self.episode_length,
        )

    def _get_demand(self, sku: str, day: int) -> float:
        """
        Calculate demand for a SKU on a given day.
        Includes base demand + variability + seasonal patterns.
        """
        config = self.sku_configs[sku]
        base = config["base_demand"]
        variability = config["variability"]

        # Random variation
        noise = np.random.normal(0, variability * base)

        # Seasonal pattern (weekend higher demand for most items)
        day_of_week = day % 7
        seasonal_factor = 1.3 if day_of_week in [5, 6] else 1.0  # Weekend boost

        # Trend: some items trending up/down
        trend = 1.0 + (0.01 * (day // 10))  # Slight uptrend every 10 days

        demand = max(0, (base + noise) * seasonal_factor * trend)
        return demand

    def _get_forecast(self) -> Dict[str, float]:
        """Get demand forecast for today."""
        forecast = {}
        for sku in self.skus:
            demand = self._get_demand(sku, self.current_day)
            forecast[sku] = max(0, demand)
        return forecast

    def _get_sales_last_7_days(self) -> Dict[str, List[float]]:
        """Get historical sales data."""
        return {sku: self.sales_history[sku][-7:] for sku in self.skus}

    def _process_orders(self, action: OrderAction) -> float:
        """
        Process agent's order action.
        Returns: total cost of orders placed
        """
        total_cost = 0.0

        for sku, quantity in action.orders.items():
            if sku not in self.skus or quantity <= 0:
                continue

            # Cost to order
            cost = quantity * self.UNIT_COSTS[sku]

            # Emergency multiplier
            if action.emergency:
                cost *= 1.5  # 50% premium
                self.emergency_orders += 1

            # Check budget
            if cost > self.budget_remaining:
                # Can't afford, skip this order
                continue

            # Add to orders in transit
            lead_time = self.sku_configs[sku]["lead_time"]
            arrival_day = self.current_day + lead_time

            self.orders_in_transit[sku][arrival_day] += quantity

            # Deduct from budget
            self.budget_remaining -= cost
            total_cost += cost

        return total_cost

    def _receive_orders(self):
        """Check if any orders arrived today."""
        for sku in self.skus:
            if self.current_day in self.orders_in_transit[sku]:
                self.inventory[sku] += self.orders_in_transit[sku][self.current_day]
                del self.orders_in_transit[sku][self.current_day]

    def _process_demand(self) -> Dict[str, float]:
        """
        Simulate customer demand.
        Returns: actual demand for each SKU
        """
        demand = {}
        for sku in self.skus:
            d = self._get_demand(sku, self.current_day)
            demand[sku] = d
        return demand

    def _calculate_reward(self, demand: Dict[str, float], order_cost: float) -> float:
        """
        Calculate reward for today's decisions.

        Reward = Revenue from sales - Ordering cost - Storage cost - Stockout penalties
        """
        revenue = 0.0
        storage_cost = 0.0
        stockout_penalty = 0.0

        for sku in self.skus:
            # Sales revenue
            sold = min(demand[sku], self.inventory[sku])
            revenue += sold * self.UNIT_PRICES[sku]

            # Update sales history
            self.sales_history[sku].append(sold)

            # Storage cost (on remaining inventory)
            remaining = self.inventory[sku] - sold
            storage_cost += remaining * self.STORAGE_COST_PER_UNIT[sku]

            # Stockout penalty
            unsold_demand = max(0, demand[sku] - self.inventory[sku])
            stockout_penalty += unsold_demand * self.UNIT_PRICES[sku] * 0.3  # 30% lost profit

            # Update inventory
            self.inventory[sku] = max(0, self.inventory[sku] - sold)

            # Track stockouts
            if self.inventory[sku] == 0 and demand[sku] > 0:
                self.stockout_days[sku] += 1

        # Calculate final reward
        profit = revenue - order_cost - storage_cost - stockout_penalty

        # Normalize to roughly 0-1 range
        max_possible_revenue = sum(
            self._get_demand(sku, self.current_day) * self.UNIT_PRICES[sku]
            for sku in self.skus
        )

        reward = profit / max_possible_revenue if max_possible_revenue > 0 else 0.0
        reward = np.clip(reward, -1.0, 1.0)

        # Update cumulative stats
        self.total_revenue += revenue
        self.total_cost += order_cost + storage_cost + stockout_penalty
        self.total_reward += reward

        return reward

    def reset(self, **kwargs) -> ShopObservation:
        """Reset episode and return initial observation."""
        self._reset_episode()
        return self._get_observation()

    def step(self, action: OrderAction, timeout_s=None, **kwargs) -> ShopObservation:
        """
        Execute one step (one day).

        Args:
            action: OrderAction with orders to place
            timeout_s: Optional timeout (not used)
            **kwargs: Additional parameters (not used)

        Returns:
            ShopObservation for the next day
        """
        # Process orders
        order_cost = self._process_orders(action)

        # Receive any orders that arrived
        self._receive_orders()

        # Process demand
        demand = self._process_demand()

        # Calculate reward
        reward = self._calculate_reward(demand, order_cost)

        # Move to next day
        self.current_day += 1
        self._state.step_count = self.current_day

        # Check if episode done
        done = self.current_day >= self.episode_length

        # Get observation
        obs = self._get_observation()
        obs.reward = reward
        obs.done = done
        obs.actual_demand = demand

        return obs

    def _get_observation(self) -> ShopObservation:
        """Get current observation."""
        forecast = self._get_forecast()

        return ShopObservation(
            inventory_levels=dict(self.inventory),
            sales_last_7_days=self._get_sales_last_7_days(),
            demand_forecast=forecast,
            actual_demand={sku: 0.0 for sku in self.skus},  # Will be updated in step()
            reorder_pending={
                sku: {day: qty for day, qty in self.orders_in_transit[sku].items()}
                for sku in self.skus
            },
            lead_times={sku: self.sku_configs[sku]["lead_time"] for sku in self.skus},
            storage_cost_per_unit={sku: self.STORAGE_COST_PER_UNIT[sku] for sku in self.skus},
            unit_cost={sku: self.UNIT_COSTS[sku] for sku in self.skus},
            unit_price={sku: self.UNIT_PRICES[sku] for sku in self.skus},
            budget_remaining=self.budget_remaining,
            supplier_min_orders={sku: 5 for sku in self.skus},
            day_of_week=self.current_day % 7,
            season=self._get_season(),
            current_day=self.current_day,
            stockout_flags={sku: self.inventory[sku] == 0 for sku in self.skus},
            stockout_days=dict(self.stockout_days),
            reward=0.0,
            done=False,
        )

    def _get_season(self) -> str:
        """Get current season based on day."""
        day_in_year = self.current_day % 365
        if day_in_year < 90:
            return "spring"
        elif day_in_year < 180:
            return "summer"
        elif day_in_year < 270:
            return "fall"
        else:
            return "winter"

    @property
    def state(self) -> ShopState:
        """Return current episode state."""
        self._state.total_revenue = self.total_revenue
        self._state.total_cost = self.total_cost
        self._state.total_profit = self.total_revenue - self.total_cost
        self._state.total_reward = self.total_reward
        self._state.stockout_days_count = sum(self.stockout_days.values())
        self._state.excess_inventory = sum(self.inventory.values())
        self._state.emergency_orders_count = self.emergency_orders
        return self._state
