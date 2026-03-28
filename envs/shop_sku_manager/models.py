# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for Shop SKU Manager Environment.

Type-safe contracts for inventory management AI agent.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class OrderAction(Action):
    """
    Action: Agent decides what SKUs to order and quantities.

    Attributes:
        orders: Dictionary of {sku_id: quantity_to_order}
                Empty dict means "don't order anything today"
        emergency: Whether to use premium emergency shipping (costs more)
    """

    orders: Dict[str, float] = Field(default_factory=dict)
    emergency: bool = Field(default=False)


class ShopObservation(Observation):
    """
    Observation: What the agent sees about shop state.

    This contains all information needed for the agent to make reorder decisions.
    """

    # Current inventory levels for each SKU
    inventory_levels: Dict[str, float] = Field(
        default_factory=dict, description="Current stock quantity per SKU"
    )

    # Demand signals
    sales_last_7_days: Dict[str, List[float]] = Field(
        default_factory=dict, description="Historical sales for each SKU (last 7 days)"
    )
    demand_forecast: Dict[str, float] = Field(
        default_factory=dict, description="Predicted demand for today"
    )
    actual_demand: Dict[str, float] = Field(
        default_factory=dict, description="What customers wanted to buy today"
    )

    # Supply chain info
    reorder_pending: Dict[str, Dict[int, float]] = Field(
        default_factory=dict,
        description="Orders in transit: {sku_id: {arrival_day: qty}}",
    )
    lead_times: Dict[str, int] = Field(
        default_factory=dict, description="Supplier lead time per SKU (days)"
    )

    # Cost info
    storage_cost_per_unit: Dict[str, float] = Field(
        default_factory=dict, description="Daily storage cost per unit"
    )
    unit_cost: Dict[str, float] = Field(
        default_factory=dict, description="Cost to buy from supplier"
    )
    unit_price: Dict[str, float] = Field(
        default_factory=dict, description="Price to sell to customers"
    )

    # Constraints
    budget_remaining: float = Field(default=0.0, description="Remaining budget for orders")
    supplier_min_orders: Dict[str, int] = Field(
        default_factory=dict, description="Minimum order quantity per SKU"
    )

    # Time & season
    day_of_week: int = Field(default=0, description="0=Monday, 6=Sunday")
    season: str = Field(default="spring", description="spring/summer/fall/winter")
    current_day: int = Field(default=0, description="Day number in episode")

    # Stockout tracking
    stockout_flags: Dict[str, bool] = Field(
        default_factory=dict, description="Whether each SKU ran out of stock"
    )
    stockout_days: Dict[str, int] = Field(
        default_factory=dict, description="Total days out of stock per SKU"
    )

    # Episode reward/status
    reward: float = Field(default=0.0, description="Reward for today's decisions")
    done: bool = Field(default=False, description="Episode finished?")


class ShopState(State):
    """
    State: Episode-level metadata.

    Tracks cumulative statistics and episode info.
    """

    episode_id: str = Field(default="", description="Unique episode ID")
    step_count: int = Field(default=0, description="Current day number")
    task_difficulty: str = Field(
        default="easy", description="easy/medium/hard"
    )

    # Cumulative metrics
    total_revenue: float = Field(default=0.0, description="Total revenue earned")
    total_cost: float = Field(default=0.0, description="Total ordering + storage cost")
    total_profit: float = Field(default=0.0, description="Revenue - Cost")
    total_reward: float = Field(default=0.0, description="Sum of all rewards")

    # Quality metrics
    stockout_days_count: int = Field(default=0, description="Total days with any stockout")
    excess_inventory: float = Field(default=0.0, description="Total excess units held")
    emergency_orders_count: int = Field(default=0, description="How many emergency orders")

    # Current episode state
    num_skus: int = Field(default=0, description="Number of SKUs in this task")
    episode_length: int = Field(default=30, description="Days per episode")
