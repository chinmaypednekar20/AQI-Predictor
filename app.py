import os
import json
import logging
from typing import Tuple, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# Import backend ML pipeline functions
from ml_pipeline import (
    KEY_POLLUTANT_FEATURES,
    TARGET_VARIABLE,
    MODELS_DIR,
    locate_or_fetch_dataset,
    load_and_preprocess_data,
    execute_pipeline
)

# --------------------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION & REFINED THEME STYLING
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="AQI Comparative Machine Learning Study",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional, human-designed data science theme (Charcoal, Royal Blue & Warm Amber)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Clean Minimalist Header (No AI tags, no glowing gradients) */
    .header-container {
        padding: 12px 0 20px 0;
        border-bottom: 1px solid #27272A;
        margin-bottom: 24px;
    }
    .header-container h1 {
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.01em;
        color: #F4F4F5;
    }
    .header-container p {
        font-size: 0.95rem;
        color: #A1A1AA;
        margin: 0;
    }

    /* Clean Streamlit Tab Underline Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #27272A;
        padding-bottom: 0px;
        margin-bottom: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        padding: 8px 16px;
        font-size: 0.95rem;
        font-weight: 500;
        color: #71717A;
        border-radius: 0;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        border-color: transparent !important;
        border-bottom: 2px solid #3B82F6 !important;
        color: #FAFAFA !important;
        font-weight: 600;
    }

    /* Clean Solid Card Box (Matte Zinc/Charcoal) */
    .card-box {
        background: #18181B;
        border: 1px solid #27272A;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }
    .card-title {
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #A1A1AA;
        margin-bottom: 6px;
    }
    .card-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #FAFAFA;
        margin: 0;
    }
    .card-subtext {
        font-size: 0.8rem;
        color: #71717A;
        margin-top: 4px;
    }

    /* Academic Comparison Badges */
    .badge-rf {
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .badge-svr {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
    }

    /* CPCB Category Indicator Pill */
    .aqi-pill {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# 2. DATA & MODEL ARTIFACT CACHING
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_dataset(filepath: str = "city_day.csv") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    clean_df, diagnostics = load_and_preprocess_data(filepath=filepath)
    return clean_df, diagnostics


@st.cache_resource(show_spinner=False)
def get_cached_pipeline(models_dir: str = MODELS_DIR) -> Dict[str, Any]:
    bundle_path = os.path.join(models_dir, "aqi_pipeline_bundle.joblib")
    if not os.path.exists(bundle_path):
        with st.spinner("Training Random Forest and SVR models on dataset..."):
            execute_pipeline(output_dir=models_dir)
    return joblib.load(bundle_path)


# --------------------------------------------------------------------------------------
# 3. CPCB NAQI CATEGORIZATION LOGIC
# --------------------------------------------------------------------------------------
def classify_cpcb_aqi(aqi_val: float) -> Dict[str, str]:
    """Classifies continuous AQI into official Central Pollution Control Board categories."""
    if aqi_val <= 50:
        return {
            'category': 'Good',
            'range': '0 - 50',
            'color': '#10B981',
            'text_color': '#FFFFFF',
            'advisory': 'Minimal health impact. Air quality is clean and satisfactory.'
        }
    elif aqi_val <= 100:
        return {
            'category': 'Satisfactory',
            'range': '51 - 100',
            'color': '#84CC16',
            'text_color': '#0F172A',
            'advisory': 'Minor breathing discomfort to sensitive individuals.'
        }
    elif aqi_val <= 200:
        return {
            'category': 'Moderate',
            'range': '101 - 200',
            'color': '#FBBF24',
            'text_color': '#0F172A',
            'advisory': 'Breathing discomfort to people with lung disease, asthma, and vulnerable individuals.'
        }
    elif aqi_val <= 300:
        return {
            'category': 'Poor',
            'range': '201 - 300',
            'color': '#F97316',
            'text_color': '#FFFFFF',
            'advisory': 'Breathing discomfort to most people on prolonged outdoor exposure.'
        }
    elif aqi_val <= 400:
        return {
            'category': 'Very Poor',
            'range': '301 - 400',
            'color': '#EF4444',
            'text_color': '#FFFFFF',
            'advisory': 'Respiratory illness on prolonged exposure. Significant distress for asthma/cardiac patients.'
        }
    else:
        return {
            'category': 'Severe',
            'range': '401 - 500+',
            'color': '#7F1D1D',
            'text_color': '#FFFFFF',
            'advisory': 'Severe respiratory risks for all individuals; serious health impacts on sensitive groups.'
        }


# --------------------------------------------------------------------------------------
# 4. INITIAL LOAD & STATE
# --------------------------------------------------------------------------------------
try:
    clean_df, diagnostics = get_cached_dataset()
    artifacts = get_cached_pipeline()
    rf_model = artifacts['rf_model']
    svr_model = artifacts['svr_model']
    scaler = artifacts['scaler']
    metrics = artifacts['metrics']
    feature_importances = artifacts['feature_importances']
    test_results_df = artifacts['test_results']
except Exception as exc:
    st.error(f"Error loading models or dataset: {exc}")
    st.stop()


# --------------------------------------------------------------------------------------
# 5. DISCREET SIDEBAR FOR SETTINGS (LEAVES MAIN UI CLEAN & PRISTINE)
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings & Dataset")
    st.write(f"**Dataset:** `city_day.csv`")
    st.write(f"**Records:** `{diagnostics['cleaned_records']:,}`")
    st.write(f"**Timespan:** `{diagnostics['date_min']}` to `{diagnostics['date_max']}`")
    st.markdown("---")
    st.caption("If you replace `city_day.csv` with a new dataset, click below to retrain both algorithms:")
    if st.button("🔄 Retrain Models with Dataset", use_container_width=True):
        with st.spinner("Retraining Random Forest and SVR..."):
            execute_pipeline()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Retraining complete!")
            st.rerun()


# --------------------------------------------------------------------------------------
# 6. CLEAN TOP HEADER (NO TAG PILLS, NO EXPANDER)
# --------------------------------------------------------------------------------------
st.markdown("""
<div class="header-container">
    <h1>Air Quality Index (AQI) Comparative Analysis</h1>
    <p>A Classical Machine Learning Study: <b>Random Forest Regressor</b> vs. <b>Support Vector Regression (SVR)</b></p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# 7. TWO CLEAN TOP TABS
# --------------------------------------------------------------------------------------
tab_eval, tab_sim = st.tabs([
    "Model Evaluation & Comparison",
    "Real-Time AQI Forecasting Simulator"
])


# ======================================================================================
# TAB 1: MODEL EVALUATION & COMPARISON
# ======================================================================================
with tab_eval:
    rf_eval = metrics['Random_Forest']
    svr_eval = metrics['Support_Vector_Regression']

    # 1. Dual Algorithm Architecture Cards
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("""
        <div class="card-box" style="border-top: 3px solid #3B82F6;">
            <div style="font-weight: 700; color: #60A5FA; font-size: 1.05rem; margin-bottom: 6px;">
                Random Forest Regressor
            </div>
            <p style="font-size: 0.88rem; color: #A1A1AA; margin: 0 0 10px 0; line-height: 1.45;">
                Fits 120 decorrelated decision trees over bootstrap samples. Uses Mean Decrease in Impurity (MDI) to capture non-linear pollutant interactions without requiring feature scaling.
            </p>
            <div style="font-size: 0.8rem; color: #71717A;">
                <b>Parameters:</b> <code>n_estimators=120, max_depth=16, min_samples_split=5</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_a2:
        st.markdown("""
        <div class="card-box" style="border-top: 3px solid #F59E0B;">
            <div style="font-weight: 700; color: #FBBF24; font-size: 1.05rem; margin-bottom: 6px;">
                Support Vector Regression (SVR)
            </div>
            <p style="font-size: 0.88rem; color: #A1A1AA; margin: 0 0 10px 0; line-height: 1.45;">
                Constructs a maximum-margin hyperplane in high-dimensional continuous space with an ε-insensitive tube. Utilizes an RBF kernel paired with standard feature scaling.
            </p>
            <div style="font-size: 0.8rem; color: #71717A;">
                <b>Parameters:</b> <code>kernel='rbf', C=100.0, epsilon=0.1, scaler=StandardScaler</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Key Performance Metrics Comparison (Test Partition)
    st.markdown("#### **Performance Benchmark** (Out-of-Sample Test Horizon: 5,472 Records)")
    m_col1, m_col2, m_col3 = st.columns(3)

    # Determine Winners
    win_rmse = "SVR" if svr_eval['rmse'] < rf_eval['rmse'] else "Random Forest"
    win_mae = "SVR" if svr_eval['mae'] < rf_eval['mae'] else "Random Forest"
    win_r2 = "SVR" if svr_eval['r2_score'] > rf_eval['r2_score'] else "Random Forest"

    with m_col1:
        badge_cls = "badge-svr" if win_rmse == "SVR" else "badge-rf"
        st.markdown(f"""
        <div class="card-box">
            <div class="card-title">Root Mean Squared Error (RMSE)</div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin: 6px 0;">
                <div>
                    <span style="color: #60A5FA; font-weight: 700; font-size: 1.4rem;">{rf_eval['rmse']:.3f}</span>
                    <span style="color: #71717A; font-size: 0.85rem; margin: 0 8px;">vs</span>
                    <span style="color: #FBBF24; font-weight: 700; font-size: 1.4rem;">{svr_eval['rmse']:.3f}</span>
                </div>
                <span class="{badge_cls}">Winner: {win_rmse}</span>
            </div>
            <div class="card-subtext">Penalizes large outlier errors (Lower is better)</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        badge_cls = "badge-svr" if win_mae == "SVR" else "badge-rf"
        st.markdown(f"""
        <div class="card-box">
            <div class="card-title">Mean Absolute Error (MAE)</div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin: 6px 0;">
                <div>
                    <span style="color: #60A5FA; font-weight: 700; font-size: 1.4rem;">{rf_eval['mae']:.3f}</span>
                    <span style="color: #71717A; font-size: 0.85rem; margin: 0 8px;">vs</span>
                    <span style="color: #FBBF24; font-weight: 700; font-size: 1.4rem;">{svr_eval['mae']:.3f}</span>
                </div>
                <span class="{badge_cls}">Winner: {win_mae}</span>
            </div>
            <div class="card-subtext">Average absolute error in AQI units (Lower is better)</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        badge_cls = "badge-svr" if win_r2 == "SVR" else "badge-rf"
        st.markdown(f"""
        <div class="card-box">
            <div class="card-title">Coefficient of Determination (R²)</div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin: 6px 0;">
                <div>
                    <span style="color: #60A5FA; font-weight: 700; font-size: 1.4rem;">{rf_eval['r2_score']:.4f}</span>
                    <span style="color: #71717A; font-size: 0.85rem; margin: 0 8px;">vs</span>
                    <span style="color: #FBBF24; font-weight: 700; font-size: 1.4rem;">{svr_eval['r2_score']:.4f}</span>
                </div>
                <span class="{badge_cls}">Winner: {win_r2}</span>
            </div>
            <div class="card-subtext">Proportion of variance explained (Higher is better)</div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Direct Side-by-Side Metric Comparison Bar Chart
    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        st.markdown("##### **Metric Comparison (Side-by-Side)**")
        metric_plot_data = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'R² (x100)'],
            'Random Forest': [rf_eval['rmse'], rf_eval['mae'], rf_eval['r2_score'] * 100],
            'Support Vector Regression': [svr_eval['rmse'], svr_eval['mae'], svr_eval['r2_score'] * 100]
        })
        fig_bar = go.Figure(data=[
            go.Bar(name='Random Forest', x=metric_plot_data['Metric'], y=metric_plot_data['Random Forest'], marker_color='#3B82F6', text=metric_plot_data['Random Forest'].round(2), textposition='auto'),
            go.Bar(name='SVR', x=metric_plot_data['Metric'], y=metric_plot_data['Support Vector Regression'], marker_color='#F59E0B', text=metric_plot_data['Support Vector Regression'].round(2), textposition='auto')
        ])
        fig_bar.update_layout(
            barmode='group',
            template="plotly_dark",
            height=300,
            margin=dict(l=30, r=20, t=20, b=20),
            paper_bgcolor="#18181B",
            plot_bgcolor="#18181B",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
        st.markdown("##### **Random Forest Feature Importance (MDI)**")
        fi_df = pd.DataFrame(list(feature_importances.items()), columns=['Pollutant', 'Weight'])
        fi_df = fi_df.sort_values(by='Weight', ascending=True)
        fig_fi = px.bar(
            fi_df,
            x='Weight',
            y='Pollutant',
            orientation='h',
            text_auto='.3f',
            color_discrete_sequence=['#3B82F6']
        )
        fig_fi.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="#18181B",
            plot_bgcolor="#18181B",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # 4. Actual vs. Predicted Time-Series Horizon
    st.markdown("---")
    st.markdown("##### **Actual vs. Predicted AQI Trajectory (Chronological Test Partition)**")
    st.caption("Chronological time-series showing true CPCB recorded AQI against Random Forest and SVR model predictions.")

    plot_df = test_results_df.copy()
    if len(plot_df) > 700:
        plot_df_sample = plot_df.iloc[::2]  # Downsample step for fast, smooth rendering
    else:
        plot_df_sample = plot_df

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=plot_df_sample['Date'],
        y=plot_df_sample['Actual_AQI'],
        name='Actual CPCB AQI',
        mode='lines',
        line=dict(color='#71717A', width=2),
        opacity=0.85
    ))
    fig_ts.add_trace(go.Scatter(
        x=plot_df_sample['Date'],
        y=plot_df_sample['RF_Predicted_AQI'],
        name='Random Forest',
        mode='lines',
        line=dict(color='#3B82F6', width=1.5)
    ))
    fig_ts.add_trace(go.Scatter(
        x=plot_df_sample['Date'],
        y=plot_df_sample['SVR_Predicted_AQI'],
        name='SVR',
        mode='lines',
        line=dict(color='#F59E0B', width=1.5, dash='dash')
    ))
    fig_ts.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor="#18181B",
        plot_bgcolor="#18181B",
        xaxis=dict(title="Chronological Test Timeline", rangeslider=dict(visible=True)),
        yaxis=dict(title="Air Quality Index (AQI)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # 5. Parity Scatter Plot
    st.markdown("---")
    st.markdown("##### **Parity Analysis (Actual vs. Predicted Alignment)**")
    sample_scatter = test_results_df.sample(n=min(500, len(test_results_df)), random_state=42)
    max_scatter_val = max(sample_scatter['Actual_AQI'].max(), sample_scatter['RF_Predicted_AQI'].max())

    fig_par = go.Figure()
    fig_par.add_trace(go.Scatter(
        x=[0, max_scatter_val], y=[0, max_scatter_val],
        mode='lines',
        name='Ideal Identity (y = x)',
        line=dict(color='#52525B', dash='dash')
    ))
    fig_par.add_trace(go.Scatter(
        x=sample_scatter['Actual_AQI'],
        y=sample_scatter['RF_Predicted_AQI'],
        mode='markers',
        name='Random Forest',
        marker=dict(color='#3B82F6', size=5, opacity=0.7)
    ))
    fig_par.add_trace(go.Scatter(
        x=sample_scatter['Actual_AQI'],
        y=sample_scatter['SVR_Predicted_AQI'],
        mode='markers',
        name='SVR',
        marker=dict(color='#F59E0B', size=5, opacity=0.7)
    ))
    fig_par.update_layout(
        template="plotly_dark",
        height=340,
        xaxis=dict(title="True Recorded AQI"),
        yaxis=dict(title="Model Predicted AQI"),
        paper_bgcolor="#18181B",
        plot_bgcolor="#18181B",
        margin=dict(l=30, r=30, t=20, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_par, use_container_width=True)


# ======================================================================================
# TAB 2: REAL-TIME AQI FORECASTING SIMULATOR
# ======================================================================================
with tab_sim:
    st.markdown("#### **Real-Time AQI Projection & Dual-Model Inference**")
    st.write(
        "Enter environmental pollutant concentrations or select a preset to generate instantaneous forecasts "
        "from both Random Forest and SVR models."
    )

    # Session state for inputs
    if 'sim_pm25' not in st.session_state:
        st.session_state.sim_pm25 = 55.0
        st.session_state.sim_pm10 = 110.0
        st.session_state.sim_no2 = 38.0
        st.session_state.sim_co = 1.2
        st.session_state.sim_o3 = 42.0

    # Atmospheric Presets
    st.markdown("###### **Quick Environmental Presets:**")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        if st.button("🌿 Clean Coastal Air", use_container_width=True):
            st.session_state.sim_pm25 = 15.0
            st.session_state.sim_pm10 = 32.0
            st.session_state.sim_no2 = 12.0
            st.session_state.sim_co = 0.5
            st.session_state.sim_o3 = 18.0
            st.rerun()
    with p_col2:
        if st.button("🏢 Typical Metro Day", use_container_width=True):
            st.session_state.sim_pm25 = 60.0
            st.session_state.sim_pm10 = 125.0
            st.session_state.sim_no2 = 42.0
            st.session_state.sim_co = 1.4
            st.session_state.sim_o3 = 45.0
            st.rerun()
    with p_col3:
        if st.button("⚠️ Poor Urban Inversion", use_container_width=True):
            st.session_state.sim_pm25 = 155.0
            st.session_state.sim_pm10 = 250.0
            st.session_state.sim_no2 = 75.0
            st.session_state.sim_co = 2.4
            st.session_state.sim_o3 = 65.0
            st.rerun()
    with p_col4:
        if st.button("🚨 Severe Winter Smog", use_container_width=True):
            st.session_state.sim_pm25 = 380.0
            st.session_state.sim_pm10 = 520.0
            st.session_state.sim_no2 = 115.0
            st.session_state.sim_co = 4.1
            st.session_state.sim_o3 = 92.0
            st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Simulator Grid: Inputs on Left, Forecasts on Right
    input_col, output_col = st.columns([1.1, 1.2], gap="large")

    with input_col:
        st.markdown("""
        <div class="card-box" style="margin-bottom: 0;">
            <div style="font-weight: 700; color: #FAFAFA; margin-bottom: 12px; font-size: 1rem;">
                Adjust Criteria Pollutants
            </div>
        """, unsafe_allow_html=True)

        val_pm25 = st.slider(
            "PM2.5 (Fine Particulate) [µg/m³]",
            min_value=0.0, max_value=500.0,
            value=float(st.session_state.sim_pm25),
            step=1.0,
            help="Fine particles from vehicle exhausts and burning."
        )
        val_pm10 = st.slider(
            "PM10 (Coarse Particulate) [µg/m³]",
            min_value=0.0, max_value=600.0,
            value=float(st.session_state.sim_pm10),
            step=1.0,
            help="Coarse inhalable dust and industrial particulate."
        )
        val_no2 = st.slider(
            "NO2 (Nitrogen Dioxide) [µg/m³]",
            min_value=0.0, max_value=250.0,
            value=float(st.session_state.sim_no2),
            step=1.0,
            help="Emissions from automobile and power plant combustion."
        )
        val_co = st.slider(
            "CO (Carbon Monoxide) [mg/m³]",
            min_value=0.0, max_value=25.0,
            value=float(st.session_state.sim_co),
            step=0.1,
            help="Combustion byproduct gas."
        )
        val_o3 = st.slider(
            "O3 (Ground Ozone) [µg/m³]",
            min_value=0.0, max_value=250.0,
            value=float(st.session_state.sim_o3),
            step=1.0,
            help="Photochemical smog reaction product."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with output_col:
        # Construct inference vector
        input_vector = pd.DataFrame([{
            'PM2.5': val_pm25,
            'PM10': val_pm10,
            'NO2': val_no2,
            'CO': val_co,
            'O3': val_o3
        }])

        # SVR requires standard scaling fitted strictly on training data
        input_vector_scaled = scaler.transform(input_vector)

        # Run model predictions
        pred_rf_val = max(0.0, float(rf_model.predict(input_vector)[0]))
        pred_svr_val = max(0.0, float(svr_model.predict(input_vector_scaled)[0]))
        consensus_aqi = (pred_rf_val + pred_svr_val) / 2.0
        cpcb_info = classify_cpcb_aqi(consensus_aqi)

        # Side-by-Side Model Projections
        pred_col1, pred_col2 = st.columns(2)
        with pred_col1:
            st.markdown(f"""
            <div class="card-box" style="border-top: 3px solid #3B82F6; padding: 14px 18px;">
                <div class="card-title">Random Forest Forecast</div>
                <div style="font-size: 1.85rem; font-weight: 700; color: #60A5FA;">{pred_rf_val:.1f}</div>
                <div class="card-subtext">Tree Ensemble Projection</div>
            </div>
            """, unsafe_allow_html=True)

        with pred_col2:
            st.markdown(f"""
            <div class="card-box" style="border-top: 3px solid #F59E0B; padding: 14px 18px;">
                <div class="card-title">SVR Model Forecast</div>
                <div style="font-size: 1.85rem; font-weight: 700; color: #FBBF24;">{pred_svr_val:.1f}</div>
                <div class="card-subtext">RBF Margin Projection</div>
            </div>
            """, unsafe_allow_html=True)

        # Consensus Ensemble Card with CPCB Health Category
        st.markdown(f"""
        <div class="card-box" style="text-align: center;">
            <div class="card-title">Consensus Predicted AQI</div>
            <div style="font-size: 3.2rem; font-weight: 800; color: #FAFAFA; margin: 4px 0;">{consensus_aqi:.1f}</div>
            <div style="margin: 12px 0;">
                <span class="aqi-pill" style="background-color: {cpcb_info['color']}; color: {cpcb_info['text_color']};">
                    {cpcb_info['category']} ({cpcb_info['range']})
                </span>
            </div>
            <p style="font-size: 0.88rem; color: #A1A1AA; margin: 10px 10px 0 10px; line-height: 1.4;">
                <b>CPCB Health Advisory:</b> {cpcb_info['advisory']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Pollutant Profile Relative Breakdown
        st.markdown("###### **Input Pollutant Profile Breakdown**")
        radar_df = pd.DataFrame({
            'Pollutant': ['PM2.5', 'PM10', 'NO2', 'CO * 10', 'O3'],
            'Concentration': [val_pm25, val_pm10, val_no2, val_co * 10, val_o3]
        })
        fig_bar_input = px.bar(
            radar_df,
            x='Pollutant',
            y='Concentration',
            color='Pollutant',
            text_auto='.1f',
            color_discrete_sequence=['#3B82F6', '#60A5FA', '#93C5FD', '#F59E0B', '#FCD34D']
        )
        fig_bar_input.update_layout(
            template="plotly_dark",
            height=200,
            showlegend=False,
            margin=dict(l=20, r=20, t=10, b=20),
            paper_bgcolor="#18181B",
            plot_bgcolor="#18181B",
            yaxis=dict(title="")
        )
        st.plotly_chart(fig_bar_input, use_container_width=True)

# --------------------------------------------------------------------------------------
# 8. CLEAN MINIMAL FOOTER
# --------------------------------------------------------------------------------------
st.markdown("""
<div style="text-align: center; color: #71717A; font-size: 0.82rem; margin-top: 36px; padding-top: 18px; border-top: 1px solid #27272A;">
    <b>Air Quality Index (AQI) Comparative Analysis</b> | Final Year Engineering Project<br>
    B.E. Computer Engineering | Student: <b>Chinmay Pednekar</b> | Random Forest vs. SVR Benchmark
</div>
""", unsafe_allow_html=True)
