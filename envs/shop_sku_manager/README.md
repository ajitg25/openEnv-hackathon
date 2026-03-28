# 🛒 Shop SKU Manager Environment

An OpenEnv environment where AI agents learn to optimize inventory management decisions for retail shops.

## Overview

**Real-world task**: Retail managers must decide:
- When to reorder products from suppliers
- How much inventory to maintain
- Whether to use emergency (premium) shipping

**Agent goal**: Maximize profit by balancing:
- ✅ Revenue from sales
- ❌ Ordering costs
- ❌ Storage costs
- ❌ Stockout penalties (lost sales)

## Quick Start

### 1. Start the Environment Server

```bash
cd /path/to/OpenEnv
PYTHONPATH=src:envs SHOP_DIFFICULTY=easy \
  uv run uvicorn envs.shop_sku_manager.server.app:app --reload
```

Server runs on `http://localhost:8000`

### 2. Connect a Client

```python
import asyncio
from envs.shop_sku_manager import ShopSKUManagerEnv
from envs.shop_sku_manager.models import OrderAction

async def main():
    env = ShopSKUManagerEnv(base_url="http://localhost:8000")

    # Reset to start new episode
    obs = await env.reset()

    # Run one episode
    while not obs.done:
        # Make ordering decision
        action = OrderAction(orders={"milk": 20, "bread": 15})

        # Execute action
        obs = await env.step(action)

        # obs contains: inventory, demand forecast, reward, done
        print(f"Reward: {obs.reward:.3f}")

    # Get episode stats
    state = await env.state()
    print(f"Final profit: ${state.total_profit:.2f}")

asyncio.run(main())
```

### 3. Run Baseline Inference

```bash
export OPENAI_API_KEY="sk-..."
python examples/shop_sku_baseline.py
```

## Tasks

### Easy: Single Product, Steady Demand
- **SKUs**: 3 (milk, bread, eggs)
- **Demand**: Constant (~20 units/day)
- **Lead time**: 2 days
- **Difficulty**: Predict simple patterns
- **Expected baseline score**: 0.30
- **Expected agent score**: 0.85

### Medium: Multiple Products, Variable Demand
- **SKUs**: 5 (+ cheese, yogurt)
- **Demand**: Varies by day (weekday/weekend patterns)
- **Lead times**: Different per SKU (1-3 days)
- **Seasonal trends**: Slight variations
- **Expected baseline score**: 0.40
- **Expected agent score**: 0.70

### Hard: Complex Multi-SKU with Constraints
- **SKUs**: 7 (+ butter, cream)
- **Demand**: Highly volatile + seasonal
- **Constraints**: Budget limits, minimum orders, emergency shipping
- **Lead times**: Variable (1-5 days)
- **Complex interactions**: Multiple SKUs compete for budget
- **Expected baseline score**: 0.20
- **Expected agent score**: 0.55

## Observation Space

Agent sees:

```python
{
    "inventory_levels": {"milk": 30.0, "bread": 25.0},
    "demand_forecast": {"milk": 20.5, "bread": 15.2},
    "sales_last_7_days": {"milk": [18, 19, 22, 20, 21, 19, 18]},
    "budget_remaining": 250.5,
    "reorder_pending": {"milk": {2: 50.0}},  # day 2: 50 units arrive
    "stockout_flags": {"milk": False, "bread": True},
    "day_of_week": 2,  # Wednesday
    "season": "spring",
    "lead_times": {"milk": 2, "bread": 1},
    "unit_price": {"milk": 4.0, "bread": 2.5},
    "reward": 0.45,
    "done": False
}
```

## Action Space

Agent decides:

```python
OrderAction(
    orders={"milk": 30, "bread": 20},  # quantities to order
    emergency=False  # use premium shipping? (50% cost increase)
)
```

## Reward Function

**Daily reward = Revenue - Costs**

```
revenue = Σ(sold_units × unit_price)
ordering_cost = Σ(ordered_qty × unit_cost × [1.5 if emergency else 1.0])
storage_cost = Σ(remaining_inventory × storage_cost_per_unit)
stockout_penalty = Σ(unsold_demand × unit_price × 0.3)

profit = revenue - ordering_cost - storage_cost - stockout_penalty
reward = profit / max_possible_revenue
```

**Dense signal**: Agent gets feedback every day, not just at episode end.

## Grading

Episodes are graded on **average daily profit**:

```python
def grade_episode(trajectory, task_difficulty):
    total_profit = sum(obs.reward for obs in trajectory)
    avg_profit = total_profit / len(trajectory)

    # Normalize by difficulty
    if task_difficulty == "easy":
        score = min(1.0, avg_profit * 1.2)  # Expected: 0.85
    elif task_difficulty == "medium":
        score = min(1.0, avg_profit * 1.4)  # Expected: 0.70
    else:  # hard
        score = min(1.0, avg_profit * 1.8)  # Expected: 0.55

    return score
```

## Environment Details

### Episode Length
30 days per episode

### Initial State
- Inventory: 30 units per SKU
- Budget: $500.00
- No pending orders

### State Dynamics

Each day:
1. Receive any orders that arrived (based on lead time)
2. Agent places new orders (within budget)
3. Customer demand occurs
4. Calculate reward (profit signal)
5. Move to next day

### Demand Generation

```
base_demand = SKU_config["base_demand"]
noise = normal(0, variability × base_demand)
seasonal = 1.3 if weekend else 1.0
trend = 1.0 + 0.01 × (day // 10)

demand = (base_demand + noise) × seasonal × trend
```

Different SKUs have different:
- Base demands
- Variability
- Lead times
- Costs

## Files

```
envs/shop_sku_manager/
├── models.py           # OrderAction, ShopObservation, ShopState
├── client.py           # ShopSKUManagerEnv client
├── openenv.yaml        # Environment specification
├── server/
│   ├── app.py          # FastAPI server
│   ├── environment.py  # Game logic
│   └── Dockerfile      # Container definition
└── README.md           # This file

examples/
└── shop_sku_baseline.py  # Baseline inference with OpenAI API
```

## Real-World Applications

This environment is useful for:
- **E-commerce**: Optimize product inventory
- **Supply chain**: Model warehouse management
- **Retail**: Train agents for store operations
- **Logistics**: Simulate delivery route decisions
- **Price optimization**: Combined with dynamic pricing

## How Agent Learns

```
Episode 1 (Random):
  Agent makes random orders
  Score: 0.30

Episode 5 (Pattern learning):
  Agent learns basic patterns
  Score: 0.45

Episode 20 (Optimization):
  Agent optimizes for forecasts
  Score: 0.70

Episode 50 (Expert):
  Agent achieves near-optimal
  Score: 0.85
```

## API Methods

### Reset

```python
obs = await env.reset(task_difficulty="easy")
# Returns: ShopObservation for day 0
```

### Step

```python
obs = await env.step(action)
# action: OrderAction
# Returns: ShopObservation for next day
```

### Get State

```python
state = await env.state()
# Returns: ShopState with cumulative stats
```

## Troubleshooting

**"Connection refused"**
- Make sure server is running in another terminal
- Check port 8000 is accessible

**"Module not found"**
- Make sure PYTHONPATH includes src and envs

**"OpenAI API key not found"**
- Export OPENAI_API_KEY before running baseline

## Implementation Notes

- **Async throughout**: Uses WebSocket for efficient communication
- **Deterministic**: Same seed produces same demand patterns
- **Scalable**: Can run multiple environments in parallel
- **Type-safe**: All models use Pydantic for validation

## Author

Ajit Gupta

## License

BSD-3-Clause
