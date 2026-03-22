"""
Orbital Insight Analytics — Streamlit Dashboard

A full-featured web dashboard for:
  1. Deforestation Detection
  2. Agricultural Productivity Prediction
  3. Urban Growth Monitoring

Run with:
    streamlit run frontend/app.py
"""

import sys
from pathlib import Path

# Allow absolute imports from the project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64

from utils.ndvi import simulate_satellite_bands, ndvi_statistics, calculate_ndvi
from utils.data_processing import (
    extract_ndvi_features,
    compute_change_metrics,
    simulate_land_use_map,
)
from utils.visualization import (
    ndvi_to_base64,
    deforestation_map_to_base64,
    urban_growth_map_to_base64,
    productivity_trend_chart,
    ndvi_histogram,
    feature_importance_chart,
    NDVI_CMAP,
)
from models import deforestation_model, agriculture_model, urban_model

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Orbital Insight Analytics",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a237e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #546e7a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f5f5f5;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #1565c0;
    }
    .alert-red {
        background: #ffebee;
        border-left: 4px solid #c62828;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        color: #b71c1c;
        font-weight: 600;
    }
    .alert-green {
        background: #e8f5e9;
        border-left: 4px solid #2e7d32;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        color: #1b5e20;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helper utilities ─────────────────────────────────────────────────────────

def show_b64_image(b64: str, caption: str = "") -> None:
    """Display a base64-encoded PNG image in Streamlit."""
    st.image(f"data:image/png;base64,{b64}", caption=caption, use_container_width=True)


def ensure_models_trained() -> None:
    """Train all models on first run (cached)."""
    with st.spinner("Training ML models (first run only) …"):
        deforestation_model.train()
        agriculture_model.train()
        urban_model.train()


@st.cache_resource(show_spinner=False)
def get_trained_deforestation_metrics() -> dict:
    return deforestation_model.train()


@st.cache_resource(show_spinner=False)
def get_trained_agriculture_metrics() -> dict:
    return agriculture_model.train()


@st.cache_resource(show_spinner=False)
def get_trained_urban_metrics() -> dict:
    return urban_model.train()


# ── Sidebar navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/The_Earth_seen_from_Apollo_17.jpg/240px-The_Earth_seen_from_Apollo_17.jpg",
        use_container_width=True,
    )
    st.title("🛰️ Orbital Insight")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "🌳 Deforestation",
            "🌾 Agriculture",
            "🏙️ Urban Growth",
            "📊 Model Metrics",
        ],
    )
    st.markdown("---")
    st.caption("Orbital Insight Analytics v1.0")


# ═══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.markdown('<div class="main-header">🛰️ Orbital Insight Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Satellite Data Analysis Platform — Deforestation · Agriculture · Urban Growth</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <h2>🌳</h2>
                <h3>Deforestation Detection</h3>
                <p>Analyse NDVI time series to detect and quantify vegetation loss using machine learning.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <h2>🌾</h2>
                <h3>Agricultural Productivity</h3>
                <p>Predict crop yield from NDVI, soil moisture, temperature, and precipitation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <h2>🏙️</h2>
                <h3>Urban Growth Monitoring</h3>
                <p>Map urban expansion between two time periods using satellite classification.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("How it works")
    st.markdown(
        """
        1. **Simulated satellite imagery** is generated using NumPy arrays representing NIR and Red bands.
        2. **NDVI** (Normalized Difference Vegetation Index) is computed:
           `NDVI = (NIR − Red) / (NIR + Red)`
        3. **Machine learning models** (Random Forest & Gradient Boosting) classify or regress on NDVI features.
        4. **Interactive visualisations** present the results as maps and charts.

        Use the sidebar to navigate to each analysis module.
        """
    )

    # Quick demo NDVI map
    st.subheader("Example NDVI Map")
    demo_bands = simulate_satellite_bands(64, 64, vegetation_fraction=0.65, seed=1)
    img_b64 = ndvi_to_base64(demo_bands["ndvi"], "Demo NDVI (Simulated)")
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        show_b64_image(img_b64)


# ═══════════════════════════════════════════════════════════════════════════════
# DEFORESTATION PAGE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🌳 Deforestation":
    st.header("🌳 Deforestation Detection")
    st.markdown(
        "Simulate NDVI before/after deforestation, visualise the change, and run the ML classifier."
    )
    st.markdown("---")

    # --- Controls ---
    with st.form("deforestation_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            veg_before = st.slider("Vegetation fraction BEFORE", 0.1, 1.0, 0.82, 0.01)
        with col2:
            veg_after = st.slider("Vegetation fraction AFTER", 0.05, 1.0, 0.28, 0.01)
        with col3:
            img_size = st.select_slider("Image size (px)", options=[32, 64, 128], value=64)
        submitted = st.form_submit_button("🔍 Analyse")

    if submitted:
        with st.spinner("Generating imagery and running model …"):
            before_bands = simulate_satellite_bands(img_size, img_size, veg_before, seed=42)
            after_bands = simulate_satellite_bands(img_size, img_size, veg_after, seed=43)
            ndvi_before = before_bands["ndvi"]
            ndvi_after = after_bands["ndvi"]

            stats_before = ndvi_statistics(ndvi_before)
            stats_after = ndvi_statistics(ndvi_after)
            change = compute_change_metrics(ndvi_before, ndvi_after)

            features = extract_ndvi_features(ndvi_after)
            features["ndvi_change"] = change["mean_change"]
            prediction = deforestation_model.predict(features)

        # --- Metrics ---
        st.subheader("NDVI Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean NDVI (before)", f"{stats_before['mean']:.3f}")
        m2.metric("Mean NDVI (after)", f"{stats_after['mean']:.3f}", delta=f"{stats_after['mean'] - stats_before['mean']:.3f}")
        m3.metric("Vegetation loss", f"{change['loss_fraction'] * 100:.1f}%")
        m4.metric("Mean NDVI change", f"{change['mean_change']:.3f}")

        # --- Prediction alert ---
        label = prediction["label"]
        prob = prediction["deforestation_probability"]
        if label == 1:
            st.markdown(
                f'<div class="alert-red">⚠️ Deforestation detected — probability: {prob:.1%}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="alert-green">✅ Vegetation intact — deforestation probability: {prob:.1%}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # --- Maps ---
        st.subheader("Satellite NDVI Maps")
        comparison_b64 = deforestation_map_to_base64(ndvi_before, ndvi_after)
        show_b64_image(comparison_b64, "Before / After / Vegetation Loss (Δ NDVI)")

        # --- Histogram ---
        st.subheader("NDVI Distribution")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.plotly_chart(ndvi_histogram(ndvi_before), use_container_width=True)
            st.caption("Before")
        with col_h2:
            st.plotly_chart(ndvi_histogram(ndvi_after), use_container_width=True)
            st.caption("After")

        # --- Raw stats ---
        with st.expander("Show detailed statistics"):
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.json({"Before": stats_before, "Change metrics": change})
            with col_s2:
                st.json({"After": stats_after, "Prediction": prediction})


# ═══════════════════════════════════════════════════════════════════════════════
# AGRICULTURE PAGE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🌾 Agriculture":
    st.header("🌾 Agricultural Productivity Prediction")
    st.markdown("Predict crop yield (tons/ha) and explore multi-year productivity trends.")
    st.markdown("---")

    tab1, tab2 = st.tabs(["Single Prediction", "Multi-Year Trend"])

    # ── Single prediction ───────────────────────────────────────────
    with tab1:
        with st.form("agr_single"):
            col1, col2 = st.columns(2)
            with col1:
                ndvi_mean = st.slider("NDVI mean", 0.0, 1.0, 0.65, 0.01)
                soil_moisture = st.slider("Soil moisture", 0.0, 1.0, 0.55, 0.01)
                temperature = st.slider("Temperature (°C)", 5.0, 45.0, 26.0, 0.5)
            with col2:
                precipitation = st.slider("Precipitation (mm)", 0, 1200, 480, 10)
                crop_type = st.selectbox("Crop type", agriculture_model.CROP_TYPES)
                days_harvest = st.slider("Days to harvest", 30, 365, 120, 5)
            predict_btn = st.form_submit_button("🌾 Predict Yield")

        if predict_btn:
            features = {
                "ndvi_mean": ndvi_mean,
                "ndvi_p50": ndvi_mean,
                "soil_moisture": soil_moisture,
                "temperature_avg": temperature,
                "precipitation_mm": precipitation,
                "crop_type": crop_type,
                "days_to_harvest": days_harvest,
            }
            result = agriculture_model.predict(features)

            st.success(
                f"Predicted yield: **{result['predicted_yield_tons_ha']} tons/ha** — {result['interpretation']}"
            )

            # Gauge chart
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=result["predicted_yield_tons_ha"],
                    gauge={
                        "axis": {"range": [0, 20]},
                        "bar": {"color": "#4CAF50"},
                        "steps": [
                            {"range": [0, 4], "color": "#ffcdd2"},
                            {"range": [4, 8], "color": "#fff9c4"},
                            {"range": [8, 20], "color": "#c8e6c9"},
                        ],
                        "threshold": {
                            "line": {"color": "#1b5e20", "width": 4},
                            "thickness": 0.75,
                            "value": 8,
                        },
                    },
                    title={"text": f"Predicted Yield — {crop_type.capitalize()} (tons/ha)"},
                )
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    # ── Multi-year trend ─────────────────────────────────────────────
    with tab2:
        st.markdown("Enter a comma-separated list of annual NDVI values (e.g. `0.4, 0.5, 0.6, 0.65, 0.7`):")
        ndvi_input = st.text_input("Annual NDVI values", value="0.40, 0.48, 0.55, 0.62, 0.68, 0.72, 0.70")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            trend_crop = st.selectbox("Crop type", agriculture_model.CROP_TYPES, key="trend_crop")
            trend_soil = st.slider("Soil moisture", 0.0, 1.0, 0.55, 0.01, key="trend_soil")
        with col_t2:
            trend_temp = st.slider("Temperature (°C)", 5.0, 45.0, 26.0, 0.5, key="trend_temp")
            trend_precip = st.slider("Precipitation (mm)", 0, 1200, 480, 10, key="trend_precip")
        start_year = st.number_input("Start year", min_value=2000, max_value=2030, value=2016, step=1)

        if st.button("📈 Generate Trend"):
            try:
                ndvi_vals = [float(v.strip()) for v in ndvi_input.split(",")]
            except ValueError:
                st.error("Please enter valid comma-separated float values.")
                st.stop()

            years = list(range(int(start_year), int(start_year) + len(ndvi_vals)))
            predictions = []
            for ndvi_val in ndvi_vals:
                feat = {
                    "ndvi_mean": ndvi_val,
                    "ndvi_p50": ndvi_val,
                    "soil_moisture": trend_soil,
                    "temperature_avg": trend_temp,
                    "precipitation_mm": trend_precip,
                    "crop_type": trend_crop,
                    "days_to_harvest": 120.0,
                }
                predictions.append(agriculture_model.predict(feat)["predicted_yield_tons_ha"])

            fig = productivity_trend_chart(years, predictions, predictions)
            fig.update_traces(name="Predicted yield", selector=dict(name="Actual"))
            fig.data[1].visible = False
            st.plotly_chart(fig, use_container_width=True)

            trend_df = pd.DataFrame(
                {"Year": years, "NDVI": ndvi_vals, "Predicted yield (t/ha)": predictions}
            )
            st.dataframe(trend_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# URBAN GROWTH PAGE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🏙️ Urban Growth":
    st.header("🏙️ Urban Growth Monitoring")
    st.markdown("Classify urban vs. non-urban land use and quantify expansion between two periods.")
    st.markdown("---")

    with st.form("urban_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            urb_before = st.slider("Urban fraction BEFORE", 0.01, 0.95, 0.15, 0.01)
        with col2:
            urb_after = st.slider("Urban fraction AFTER", 0.01, 0.95, 0.38, 0.01)
        with col3:
            urb_size = st.select_slider("Image size (px)", options=[32, 64, 128], value=64)
        run_urban = st.form_submit_button("🏙️ Analyse Urban Growth")

    if run_urban:
        with st.spinner("Running urban classification …"):
            size = urb_size
            before_bands = simulate_satellite_bands(
                size, size, vegetation_fraction=1.0 - urb_before, seed=10
            )
            after_bands = simulate_satellite_bands(
                size, size, vegetation_fraction=1.0 - urb_after, seed=11
            )

            before_map = urban_model.predict_map(
                before_bands["ndvi"], before_bands["red"], before_bands["nir"]
            )
            after_map = urban_model.predict_map(
                after_bands["ndvi"], after_bands["red"], after_bands["nir"]
            )

            stats = urban_model.compute_urban_growth_stats(before_map, after_map)

        # --- Metrics ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Urban BEFORE", f"{stats['urban_before_pct']:.1f}%")
        m2.metric("Urban AFTER", f"{stats['urban_after_pct']:.1f}%", delta=f"{stats['urban_after_pct'] - stats['urban_before_pct']:.1f}%")
        m3.metric("New urban pixels", stats["new_urban_pixels"])
        m4.metric("Urban growth", f"{stats['growth_pct']:.1f}%")

        # --- Maps ---
        st.subheader("Urban Classification Maps")
        img_b64 = urban_growth_map_to_base64(before_map, after_map)
        show_b64_image(img_b64, "Urban land use: Before / After / New Urban Areas")

        # --- Bar chart ---
        st.subheader("Urban Area Comparison")
        bar_fig = go.Figure(
            [
                go.Bar(
                    name="Urban %",
                    x=["Before", "After"],
                    y=[stats["urban_before_pct"], stats["urban_after_pct"]],
                    marker_color=["#90caf9", "#1565c0"],
                )
            ]
        )
        bar_fig.update_layout(
            yaxis_title="Urban Area (%)",
            template="plotly_white",
            title="Urban Coverage Before vs After",
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        with st.expander("Show raw statistics"):
            st.json(stats)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL METRICS PAGE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Model Metrics":
    st.header("📊 Model Performance Metrics")
    st.markdown("Training metrics and feature importances for each ML model.")
    st.markdown("---")

    tab_def, tab_agr, tab_urb = st.tabs(
        ["🌳 Deforestation", "🌾 Agriculture", "🏙️ Urban"]
    )

    # ── Deforestation ────────────────────────────────────────────────
    with tab_def:
        with st.spinner("Loading deforestation model metrics …"):
            metrics_def = get_trained_deforestation_metrics()

        st.metric("Accuracy", f"{metrics_def['accuracy']:.4f}")

        st.subheader("Feature Importances")
        fig_fi = feature_importance_chart(
            metrics_def["feature_names"], metrics_def["feature_importances"]
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        st.subheader("Confusion Matrix")
        cm = np.array(metrics_def["confusion_matrix"])
        cm_fig = px.imshow(
            cm,
            text_auto=True,
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            x=["Intact", "Deforested"],
            y=["Intact", "Deforested"],
            color_continuous_scale="Blues",
        )
        cm_fig.update_layout(title="Confusion Matrix — Deforestation Classifier")
        st.plotly_chart(cm_fig, use_container_width=True)

        with st.expander("Classification Report"):
            rep = metrics_def["classification_report"]
            st.json(rep)

    # ── Agriculture ──────────────────────────────────────────────────
    with tab_agr:
        with st.spinner("Loading agriculture model metrics …"):
            metrics_agr = get_trained_agriculture_metrics()

        col_a1, col_a2, col_a3 = st.columns(3)
        col_a1.metric("RMSE (tons/ha)", f"{metrics_agr['rmse']:.4f}")
        col_a2.metric("MAE (tons/ha)", f"{metrics_agr['mae']:.4f}")
        col_a3.metric("R²", f"{metrics_agr['r2']:.4f}")

        st.subheader("Feature Importances")
        fig_fi_agr = feature_importance_chart(
            metrics_agr["feature_names"], metrics_agr["feature_importances"]
        )
        st.plotly_chart(fig_fi_agr, use_container_width=True)

    # ── Urban ────────────────────────────────────────────────────────
    with tab_urb:
        with st.spinner("Loading urban model metrics …"):
            metrics_urb = get_trained_urban_metrics()

        st.metric("Accuracy", f"{metrics_urb['accuracy']:.4f}")

        st.subheader("Feature Importances")
        fig_fi_urb = feature_importance_chart(
            metrics_urb["feature_names"], metrics_urb["feature_importances"]
        )
        st.plotly_chart(fig_fi_urb, use_container_width=True)

        with st.expander("Classification Report"):
            st.json(metrics_urb["classification_report"])
