"""Streamlit dashboard for *completed* offline runs; no live decisioning claims."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Automotive Analytics", page_icon="📊", layout="wide")
st.title("Automotive Analytics & Model Quality")
st.caption("Offline demonstration · 500-record public dataset · not validated for commercial pricing")
folder = Path(st.sidebar.text_input("Output directory", "outputs"))
required = [folder / n for n in ["metrics.json", "data_quality.json", "holdout_predictions.csv", "segments.csv"]]
if not all(p.is_file() for p in required):
    st.info("Run the pipeline first: python -m automotive_ai.pipeline --data data/Car_Purchasing_Data.csv --out outputs")
    st.stop()
metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
quality = json.loads((folder / "data_quality.json").read_text(encoding="utf-8"))
holdout = pd.read_csv(folder / "holdout_predictions.csv")
segments = pd.read_csv(folder / "segments.csv")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Holdout R²", f"{metrics['r2']:.3f}")
col2.metric("Holdout MAE", f"{metrics['mae']:,.0f}")
col3.metric("Valid records", quality["valid_rows"])
col4.metric("Rejected records", quality["rejected_rows"])
tab1, tab2, tab3 = st.tabs(["Prediction quality", "Customer segments", "Data quality"])
with tab1:
    st.subheader("Actual versus predicted budgets")
    st.scatter_chart(holdout, x="actual_budget", y="predicted_budget")
    st.subheader("Absolute errors by customer segment")
    st.bar_chart(holdout.groupby("cluster")["absolute_error"].mean())
    st.dataframe(holdout, use_container_width=True)
with tab2:
    st.subheader("Train-set customer segments")
    st.bar_chart(segments.set_index("cluster")["customers"])
    st.dataframe(segments, use_container_width=True)
with tab3:
    st.json(quality)
    st.caption("Deduplication, numeric parsing, range checks, missingness and row rejection are reported.")

with st.expander("Vehicle recommendation and simulated pricing (synthetic)", expanded=False):
    st.caption("Demonstration only: illustrative inventory, assumed purchase probability, and offline Q-learning. Not for real commercial pricing.")
    import numpy as np
    from automotive_ai.decision_support import recommend_vehicles, simulate_pricing
    budget = st.number_input("Customer budget (illustrative currency units)", min_value=1000.0, value=40000.0, step=1000.0)
    options = recommend_vehicles(budget)
    st.subheader("Closest vehicles by list price (KNN)")
    st.dataframe(options, use_container_width=True, hide_index=True)
    chosen = st.selectbox("Vehicle for simulated pricing", options['model'].tolist())
    price = float(options.loc[options['model'] == chosen, 'list_price'].iloc[0])
    table, policy = simulate_pricing(price, budget)
    st.subheader("Discount policy experiment (Q-learning)")
    st.metric("Learned discount (synthetic)", f"{policy['recommended_discount_pct']}%")
    st.subheader("Pricing Policy Evaluation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Learned Discount",
            f"{policy['recommended_discount_pct']}%"
        )

    with col2:
        st.metric(
            "Theoretical Best Discount",
            f"{policy['theoretical_best_discount_pct']}%"
        )

    with col3:
        st.metric(
            "Optimality Gap",
            f"{policy['optimality_gap']:,.2f}"
        )

    st.caption(
        "Results are based on a synthetic purchase-probability "
        "model, not real customer transactions."
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption("Learned action depends on simulation assumptions and random seed; not a validated sales uplift.")
