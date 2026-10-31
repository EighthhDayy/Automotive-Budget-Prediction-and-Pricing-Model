import numpy as np
import pandas as pd
import pytest
from automotive_ai.pipeline import FEATURES, TARGET, validate_data, train, predict_one


def test_missing_column():
    with pytest.raises(ValueError, match="Missing columns"):
        validate_data(pd.DataFrame({"Age": [30]}))


def test_data_quality_rejects_invalid_rows():
    df = pd.DataFrame({
        "Age": [25, 25, 300, 40], "Annual Salary": [40000]*4,
        "Net Worth": [10000]*4, "Credit Card Debt": [500]*4,
        TARGET: [20000, 20000, 50000, -1],
    })
    with pytest.raises(ValueError, match="minimum 30"):
        validate_data(df)


def test_end_to_end(tmp_path):
    rng = np.random.default_rng(11)
    n = 100
    age = rng.integers(22, 65, n)
    salary = rng.uniform(30000, 170000, n)
    networth = rng.uniform(5000, 700000, n)
    debt = rng.uniform(0, 50000, n)
    frame = pd.DataFrame({"Age": age, "Annual Salary": salary,
                          "Net Worth": networth, "Credit Card Debt": debt,
                          TARGET: 9000 + age*80 + salary*.15 + networth*.012 - debt*.02})
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    result = train(path, tmp_path / "outputs")
    assert result["metrics"]["test_rows"] == 20
    assert all((tmp_path / "outputs" / f).exists() for f in
               ["models.joblib", "metrics.json", "data_quality.json", "segments.csv", "holdout_predictions.csv"])
    pred = predict_one(tmp_path / "outputs" / "models.joblib", 30, 80000, 100000, 2000)
    assert pred["predicted_budget"] > 0
    assert pred["segment"] in [0, 1, 2]
