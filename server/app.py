"""
Server entry point for Shop SKU Manager environment.

Re-exports the FastAPI app from the environment package
so the evaluator can find it at server/app.py.
"""

import sys
from pathlib import Path

# Ensure envs is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "envs"))

from shop_sku_manager.server.app import app  # noqa: F401, E402

__all__ = ["app"]
