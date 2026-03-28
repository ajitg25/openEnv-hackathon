# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for Shop SKU Manager Environment.

Exposes the inventory management environment over HTTP/WebSocket.
"""

import os
from ..models import OrderAction, ShopObservation
from .environment import ShopSKUManagerEnvironment

# Import OpenEnv's HTTP server creator
try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    from openenv.core.env_server.http_server import create_app


def create_environment_factory(difficulty: str = "easy"):
    """Factory function to create new environment instances."""
    def factory():
        return ShopSKUManagerEnvironment(
            task_difficulty=difficulty,
            episode_length=30,
            initial_budget=500.0,
        )
    return factory


# Get difficulty from environment variable
difficulty = os.getenv("SHOP_DIFFICULTY", "easy")

# Create the FastAPI app
app = create_app(
    create_environment_factory(difficulty),
    OrderAction,
    ShopObservation,
    env_name="shop_sku_manager",
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
