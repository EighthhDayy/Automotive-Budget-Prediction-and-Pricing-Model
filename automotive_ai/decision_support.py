"""Transparent, offline vehicle recommendations and simulated pricing decisions.

All vehicles and outcomes are synthetic. No real sales or conversion claims.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

DEFAULT_INVENTORY = pd.DataFrame([
    ("A100", 18000), ("A150", 20000), ("A200", 22000),
    ("A250", 24000), ("A300", 27000), ("M100", 32000),
    ("M200", 38000), ("M300", 44000), ("X100", 52000),
    ("X200", 60000), ("X300", 70000),
], columns=["model", "list_price"])


def recommend_vehicles(budget: float, inventory: pd.DataFrame | None = None,
                       k: int = 3) -> pd.DataFrame:
    """KNN recommendations by absolute price distance, not personalization."""
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("Budget must be finite and positive")
    if k < 1:
        raise ValueError("k must be positive")
    vehicles = DEFAULT_INVENTORY.copy() if inventory is None else inventory.copy()
    if not {'model', 'list_price'}.issubset(vehicles.columns):
        raise ValueError("Inventory requires model and list_price")
    vehicles['list_price'] = pd.to_numeric(vehicles['list_price'], errors='coerce')
    if vehicles.empty or vehicles['list_price'].isna().any() or (vehicles['list_price'] <= 0).any():
        raise ValueError("Inventory prices must be positive numbers")
    nn = NearestNeighbors(n_neighbors=min(k, len(vehicles)), metric='euclidean')
    nn.fit(vehicles[['list_price']])
    distances, indices = nn.kneighbors(pd.DataFrame({'list_price': [budget]}))
    result = vehicles.iloc[indices[0]].copy().reset_index(drop=True)
    result['distance_from_budget'] = distances[0]
    result['within_budget'] = result['list_price'] <= budget
    return result


@dataclass(frozen=True)
class PricingConfig:
    discount_rates: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15)
    cost_fraction: float = 0.72
    episodes: int = 2000
    learning_rate: float = 0.12
    exploration_rate: float = 0.2
    seed: int = 42


def _purchase_probability(offer: float, budget: float) -> float:
    """Assumed synthetic willingness-to-buy curve, not estimated from customers."""
    ratio = offer / budget
    return float(np.clip(0.5 - (ratio - 1.0) * 2.0, 0.02, 0.98))


def simulate_pricing(list_price: float, budget: float,
                     config: PricingConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """Multi-armed Q-learning (one state) using simulated conversion and margin rewards.

    Actions are discounts. Reward = realized gross margin if synthetic purchase occurs,
    otherwise zero. This is an offline demonstration, not an optimized live strategy.
    """
    cfg = PricingConfig() if config is None else config
    if not (np.isfinite(list_price) and np.isfinite(budget) and list_price > 0 and budget > 0):
        raise ValueError("Prices and budget must be finite positive numbers")
    if cfg.episodes < 1 or not (0 < cfg.learning_rate <= 1) or not (0 <= cfg.exploration_rate <= 1):
        raise ValueError("Invalid training configuration")
    if not (0 < cfg.cost_fraction < 1) or not cfg.discount_rates or any(not (0 <= d < 1) for d in cfg.discount_rates):
        raise ValueError("Invalid pricing configuration")
    rng = np.random.default_rng(cfg.seed)
    q = np.zeros(len(cfg.discount_rates), dtype=float)
    counts = np.zeros(len(q), dtype=int)
    for _ in range(cfg.episodes):
        if rng.random() < cfg.exploration_rate:
            idx = int(rng.integers(0, len(q)))
        else:
            idx = int(np.argmax(q))
        discount = cfg.discount_rates[idx]
        offer = list_price * (1 - discount)
        probability = _purchase_probability(offer, budget)
        margin = max(offer - list_price * cfg.cost_fraction, 0)
        reward = margin if rng.random() < probability else 0.0
        counts[idx] += 1
        q[idx] += (reward - q[idx]) / counts[idx]
    rows = []
    for idx, rate in enumerate(cfg.discount_rates):
        offer = list_price * (1 - rate)
        probability = _purchase_probability(offer, budget)
        margin = max(offer - list_price * cfg.cost_fraction, 0)
        rows.append({'discount_pct': int(round(rate * 100)), 'offer_price': round(offer, 2),
                     'assumed_conversion_probability': round(probability, 4),
                     'expected_margin_per_opportunity': round(probability * margin, 2),
                     'learned_q_value': round(float(q[idx]), 2), 'training_action_count': int(counts[idx])})
    table = pd.DataFrame(rows)
    # 1. Bandit 学习得到的策略
    chosen = int(np.argmax(q))

    # 2. 根据模拟环境计算理论最优策略
    optimal_idx = int(
        table["expected_margin_per_opportunity"].idxmax()
    )

    # 3. 计算两种策略的理论预期利润
    learned_expected_margin = float(
        table.loc[chosen, "expected_margin_per_opportunity"]
    )

    optimal_expected_margin = float(
        table.loc[optimal_idx, "expected_margin_per_opportunity"]
    )

    # 4. 计算预期利润差距
    optimality_gap = max(
        optimal_expected_margin - learned_expected_margin,
        0.0
    )

    return table, {
        "recommended_discount_pct": int(
            round(cfg.discount_rates[chosen] * 100)
        ),
        "recommended_offer_price": float(
            round(list_price * (1 - cfg.discount_rates[chosen]), 2)
        ),
        "theoretical_best_discount_pct": int(
            table.loc[optimal_idx, "discount_pct"]
        ),
        "learned_expected_margin": learned_expected_margin,
        "theoretical_best_expected_margin": optimal_expected_margin,
        "optimality_gap": round(optimality_gap, 2),
        "episodes": cfg.episodes,
        "seed": cfg.seed,
        "assumption": (
            "Synthetic purchase probability; "
            "no historical conversion validation"
        )
    }
