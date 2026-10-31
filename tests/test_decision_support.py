import pandas as pd
import pytest
from automotive_ai.decision_support import recommend_vehicles, simulate_pricing, PricingConfig


def test_knn_nearest_and_bounds():
    sample = pd.DataFrame({'model': ['A', 'B', 'C'], 'list_price': [10000, 20000, 30000]})
    r = recommend_vehicles(21000, sample, 2)
    assert r['model'].tolist() == ['B', 'C']
    assert r['within_budget'].tolist() == [True, False]
    with pytest.raises(ValueError):
        recommend_vehicles(-1)


def test_pricing_reproducible_and_valid():
    a, pa = simulate_pricing(40000, 38000, PricingConfig(episodes=300, seed=5))
    b, pb = simulate_pricing(40000, 38000, PricingConfig(episodes=300, seed=5))
    pd.testing.assert_frame_equal(a, b)
    assert pa == pb
    assert a['training_action_count'].sum() == 300
    assert a['assumed_conversion_probability'].between(0, 1).all()
    assert pa['recommended_discount_pct'] in (0, 5, 10, 15)


def test_pricing_rejects_invalid_input():
    with pytest.raises(ValueError):
        simulate_pricing(0, 30000)


import pytest
from automotive_ai.decision_support import simulate_pricing


def test_pricing_policy_evaluation():
    """Validate learned policy against the theoretical benchmark."""

    table, policy = simulate_pricing(
        list_price=32000,
        budget=30000
    )

    # 理论最优折扣应为 5%
    assert policy["theoretical_best_discount_pct"] == 5

    # 理论最优预期利润应为 3483.73
    assert policy["theoretical_best_expected_margin"] == pytest.approx(
        3483.73,
        abs=0.01
    )

    # 验证学习策略与理论最优策略的利润差距
    expected_gap = (
        policy["theoretical_best_expected_margin"]
        - policy["learned_expected_margin"]
    )

    assert policy["optimality_gap"] == pytest.approx(
        expected_gap,
        abs=0.01
    )

    # 所有报价必须符合对应折扣率
    for _, row in table.iterrows():
        expected_offer = 32000 * (
            1 - row["discount_pct"] / 100
        )

        assert row["offer_price"] == pytest.approx(
            expected_offer,
            abs=0.01
        )
