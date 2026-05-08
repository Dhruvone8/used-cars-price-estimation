import streamlit as st
import json
import numpy as np
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from predict import load_artifacts, prepare_input, predict_price, predict_range
from explainer import get_shap_explainer, get_shap_values, explain

# ── Page config ──────────────────────────────────────
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
)

# ── Load artifacts ───────────────────────────────────
@st.cache_resource
def load_all():
    model_dir = os.path.join(BASE_DIR, "models")
    ml_model, encoders = load_artifacts(
        os.path.join(model_dir, 'car_price_model.pkl'),
        os.path.join(model_dir, 'encoders.pkl'),
    )
    with open(os.path.join(model_dir, 'brand_model_map.json')) as f:
        brand_model_map = json.load(f)
    shap_explainer = get_shap_explainer(ml_model)
    return ml_model, encoders, brand_model_map, shap_explainer


ml_model, encoders, brand_model_map, shap_explainer = load_all()

# ── Header ───────────────────────────────────────────
st.title("🚗 Used Car Price Predictor — India")
st.caption("Real market data · XGBoost + SHAP · 11 cities · 4,200+ listings")
st.divider()

# ── Input Form ───────────────────────────────────────
st.subheader("📋 Car Details")

col1, col2, col3 = st.columns(3)

with col1:
    brand = st.selectbox("Brand", sorted(brand_model_map.keys()))
    model_base = st.selectbox("Model", sorted(brand_model_map[brand]))
    year  = st.slider("Purchase Year", 2004, 2024, 2018)
    km    = st.number_input("Kilometres Driven", 500, 500000, 45000, step=1000,
                             help="Total distance the car has been driven")

with col2:
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    transmission = st.radio("Transmission", ["Manual", "Automatic"], horizontal=True)
    owner_label  = st.radio(
        "Ownership",
        ["1st Owner", "2nd Owner", "3rd Owner"],
        horizontal=True,
        help="How many people have owned this car including the current seller"
    )
    owner_val = {"1st Owner": 1, "2nd Owner": 2, "3rd Owner": 3}[owner_label]

with col3:
    location = st.selectbox(
        "City",
        sorted(encoders['Location'].classes_),
        help="City where the car is being bought or sold"
    )
    engine = st.number_input(
        "Engine Displacement (cc)",
        min_value=600, max_value=5000, value=1200, step=50,
        help="Engine size in cubic centimetres — find this in your RC book or car manual"
    )
    seats = st.radio(
        "Seating Capacity",
        [4, 5, 7, 8],
        index=1,
        horizontal=True
    )

st.divider()

# ── Predict button ───────────────────────────────────
predict_btn = st.button("🔍 Estimate Price", width='stretch', type="primary")

if predict_btn:
    raw = {
        'Brand':        brand,
        'Model_Base':   model_base,
        'Year':         year,
        'Km':           km,
        'Fuel':         fuel,
        'Transmission': transmission,
        'Owner':        owner_val,
        'Location':     location,
        'Engine':       engine,
        'Seats':        float(seats),
    }

    with st.spinner("Analysing market data..."):
        input_df   = prepare_input(raw, encoders)
        predicted  = predict_price(ml_model, input_df)
        low, high  = predict_range(predicted)

        input_array          = input_df.values.astype(float)
        shap_vals, base_price = get_shap_values(shap_explainer, input_array)

        result = explain(
            shap_vals       = shap_vals,
            feature_names   = input_df.columns.tolist(),
            feature_vals    = input_df.values[0],
            predicted_price = predicted,
            base_price      = base_price,
        )

    # ── Price display ────────────────────────────────
    st.subheader("💰 Estimated Market Price")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Conservative",   f"₹{low:,.0f}",       help="Lower bound — expect this if negotiated down")
    m2.metric("Fair Value",     f"₹{predicted:,.0f}", help="Model's best estimate of fair market price")
    m3.metric("Optimistic",     f"₹{high:,.0f}",      help="Upper bound — if seller holds firm")
    m4.metric("Deal Score",     f"{result['deal_score']} / 10",
              help="10 = excellent value factors, 1 = mostly value-reducing factors")

    # Price range bar
    st.progress(
        result['deal_score'] / 10,
        text=f"Value score: {result['deal_score']}/10 — {result['summary']}"
    )

    # ── Warnings first ───────────────────────────────
    for w in result['warnings']:
        st.warning(w)

    st.divider()

    # ── Price breakdown ──────────────────────────────
    st.subheader("📊 Price Breakdown — What's driving this estimate?")
    st.caption(
        "Each factor below shows how much it pushes the price up or down "
        "compared to the average car in our dataset."
    )

    if not result['price_breakdown']:
        st.info("No significant factors to display for this car.")
    else:
        for item in result['price_breakdown']:
            impact_color = "🟢" if item['direction'] == 'positive' else "🔴" if item['impact_rs'] > 50000 else "🟡"
            with st.expander(
                f"{item['icon']}  **{item['label']}** — {item['impact_label']}",
                expanded=True
            ):
                # Simple explanation
                st.markdown(f"**What this means for you:** {item['simple']}")

                # Visual impact bar
                bar_val = min(item['impact_rs'] / 300000, 1.0)
                if item['direction'] == 'positive':
                    st.success(f"Impact: {item['impact_label']}")
                else:
                    st.error(f"Impact: {item['impact_label']}")

                # Detailed explanation in smaller text
                st.caption(f"📖 More detail: {item['detail']}")

    st.divider()

    # ── Buyer / Seller tips ───────────────────────────
    breakdown  = result['price_breakdown']
    negatives  = [b for b in breakdown if b['direction'] == 'negative']
    positives  = [b for b in breakdown if b['direction'] == 'positive']

    tip_col1, tip_col2 = st.columns(2)

    with tip_col1:
        st.subheader("🛒 Buyer Tips")
        if negatives:
            st.markdown("Use these factors to **negotiate the price down:**")
            for n in negatives[:3]:
                st.markdown(f"- **{n['label']}** reduces value by {n['impact_label'].replace('-', '')} — point this out to the seller.")
        else:
            st.markdown("This car has strong value factors — less room to negotiate down.")

    with tip_col2:
        st.subheader("💼 Seller Tips")
        if positives:
            st.markdown("Highlight these factors to **justify your asking price:**")
            for p in positives[:3]:
                st.markdown(f"- **{p['label']}** adds {p['impact_label']} to value — emphasise this to buyers.")
        else:
            st.markdown("Consider pricing conservatively — most factors are working against the asking price.")

    st.divider()

    # ── Technical SHAP details ───────────────────────
    with st.expander("🔬 Technical Details — SHAP Feature Importance"):
        st.caption(
            "SHAP (SHapley Additive exPlanations) measures each feature's individual "
            "contribution to the predicted log(price). Positive = pushes price up, "
            "Negative = pushes price down."
        )

        if result['tech_table']:
            tech_df = pd.DataFrame(result['tech_table'])
            st.dataframe(tech_df, width='stretch', hide_index=True)

        # SHAP bar chart
        shap_chart_df = pd.DataFrame({
            'Feature': input_df.columns.tolist(),
            'SHAP Value': shap_vals,
        }).sort_values('SHAP Value', key=abs, ascending=True)

        st.bar_chart(
            shap_chart_df.set_index('Feature')['SHAP Value'],
            width='stretch',
        )

        st.caption(
            f"Base price (average car in dataset): ₹{base_price:,.0f} | "
            f"Predicted: ₹{predicted:,.0f}"
        )

    # ── Model info ───────────────────────────────────
    with st.expander("ℹ️ About this prediction"):
        st.markdown("""
        **Model:** XGBoost Regressor, tuned with RandomizedSearchCV (50 iterations, 5-fold CV)

        **Test set performance:**
        | Metric | Value |
        |--------|-------|
        | R² Score | 0.9519 — explains 95.2% of price variance |
        | MAPE | 10.31% — average prediction error |
        | MAE | ₹86,446 — mean absolute error |
        | RMSE | ₹1,73,132 |

        **Dataset:** ~4,200 used car listings · 12 brands · 31 models · 11 Indian cities · 2004–2022

        **Data sources:**
        - Live scraped from CarDekho API (May 2026)
        - Kaggle CarDekho dataset (historical)

        **Limitations:**
        - Limited data for 2021+ vehicles — predictions may underestimate recent car prices
        - Model trained on listed prices, not final transaction prices
        - Predictions are estimates — always verify with live listings on CarDekho or Cars24
        """)

# ── Footer ───────────────────────────────────────────
st.divider()
st.caption(
    "Built with XGBoost + SHAP · Data from CarDekho · "
    "For informational purposes only · Not financial advice"
)