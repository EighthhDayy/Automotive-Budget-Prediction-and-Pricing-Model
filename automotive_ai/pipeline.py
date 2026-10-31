"""Reproducible automotive data quality, training and analytics workflow."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

FEATURES = ["Age", "Annual Salary", "Net Worth", "Credit Card Debt"]
TARGET = "Car Purchase Amount"
REQUIRED = FEATURES + [TARGET]


def validate_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reject unusable records and report measurable quality issues."""
    absent = [c for c in REQUIRED if c not in df.columns]
    if absent:
        raise ValueError(f"Missing columns: {absent}")
    work = df.copy()
    issues = {"input_rows": int(len(work)), "duplicate_rows": int(work.duplicated().sum())}
    work = work.drop_duplicates()
    for col in REQUIRED:
        original = work[col].copy()
        work[col] = pd.to_numeric(work[col], errors="coerce")
        issues[f"invalid_numeric_{col}"] = int((original.notna() & work[col].isna()).sum())
    issues["missing_features"] = {c: int(work[c].isna().sum()) for c in FEATURES}
    invalid = (work["Age"].notna() & ~work["Age"].between(16, 110))
    for col in ["Annual Salary", "Net Worth", "Credit Card Debt", TARGET]:
        invalid |= (work[col].notna() & (work[col] < 0))
    invalid |= work[TARGET].isna()
    issues["rejected_rows"] = int(invalid.sum())
    work = work.loc[~invalid].copy()
    issues["valid_rows"] = int(len(work))
    issues["target_missing_or_invalid"] = int(df[TARGET].isna().sum()) if TARGET in df else 0
    if len(work) < 30:
        raise ValueError("Not enough valid rows (minimum 30).")
    return work, issues


def train(csv_path: str | Path, out_dir: str | Path, seed: int = 42) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    clean, quality = validate_data(df)
    # Holdout is created BEFORE all fitted preprocessing and clustering.
    X_train, X_test, y_train, y_test = train_test_split(
        clean[FEATURES], clean[TARGET], test_size=0.2, random_state=seed
    )
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([("num", numeric, FEATURES)])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", XGBRegressor(n_estimators=160, max_depth=3, learning_rate=0.05,
                                   subsample=0.9, colsample_bytree=0.9,
                                   random_state=seed, n_jobs=1, objective="reg:squarederror"))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "r2": round(float(r2_score(y_test, predictions)), 5),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 3),
        "train_rows": len(X_train), "test_rows": len(X_test),
        "seed": seed, "target": TARGET, "features": FEATURES,
        "dataset_source": str(csv_path),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Segment the training population only; labels are descriptive, not ground truth.
    segmentation_preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    train_scaled = segmentation_preprocess.fit_transform(X_train)
    clusterer = KMeans(n_clusters=3, random_state=seed, n_init=10)
    clusters = clusterer.fit_predict(train_scaled)
    profiles = X_train.copy()
    profiles["cluster"] = clusters
    segment_summary = profiles.groupby("cluster", as_index=False).agg(
        customers=("Age", "size"), mean_age=("Age", "mean"),
        mean_salary=("Annual Salary", "mean"), mean_net_worth=("Net Worth", "mean")
    )
    segment_summary.to_csv(out / "segments.csv", index=False)
    test_results = X_test.copy()
    test_results["actual_budget"] = y_test.to_numpy()
    test_results["predicted_budget"] = predictions
    test_results["absolute_error"] = abs(y_test.to_numpy() - predictions)
    test_results["cluster"] = clusterer.predict(segmentation_preprocess.transform(X_test))
    test_results.to_csv(out / "holdout_predictions.csv", index=False)
    joblib.dump({"model": model, "segmentation_preprocess": segmentation_preprocess,
                 "clusterer": clusterer, "features": FEATURES}, out / "models.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    return {"metrics": metrics, "quality": quality}


def predict_one(model_path: str | Path, age: float, annual_salary: float,
                net_worth: float, credit_card_debt: float) -> dict:
    artifact = joblib.load(model_path)  # Only load trusted locally generated artifacts.
    sample = pd.DataFrame([[age, annual_salary, net_worth, credit_card_debt]], columns=FEATURES)
    clean_values = artifact["segmentation_preprocess"].transform(sample)
    return {"predicted_budget": float(artifact["model"].predict(sample)[0]),
            "segment": int(artifact["clusterer"].predict(clean_values)[0])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automotive data/ML pipeline")
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.out, args.seed), indent=2))


if __name__ == "__main__":
    main()
