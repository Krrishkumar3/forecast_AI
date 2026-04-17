"""
dashboard.py — Interactive Streamlit Dashboard for NatWest Forecasting.

A single-page application that lets users:
  1. Upload a CSV or use the bundled sample dataset
  2. Select a forecast horizon (1–6 weeks) and view line charts with
     historical data, baseline, likely estimate, and shaded uncertainty bounds
  3. Browse detected anomalies with AI-generated explanations
  4. Adjust a "What-if" slider and see scenario projections update in real-time
  5. Track goal probability with statistical confidence
  6. Run walk-forward backtesting to validate model accuracy
  7. Decompose trends into components and analyse autocorrelation
  8. Compare multiple forecasting models head-to-head
  9. View automated data quality health checks

Run with:
    streamlit run src/dashboard.py
"""

import sys
import os

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from scipy.stats import norm

from src.core.forecaster import Forecaster
from src.core.anomaly_detector import AnomalyDetector
from src.core.scenario_runner import ScenarioForecaster
from src.core.explainer import AnomalyExplainer
from src.core.trend_analyzer import TrendAnalyzer
from src.core.backtester import Backtester


# =====================================================================
# Page configuration
# =====================================================================
st.set_page_config(
    page_title="NatWest Advanced Forecasting",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# Custom CSS for a polished, premium look
# =====================================================================
st.markdown("""
<style>
    /* ---- Global ---- */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%);
        border-right: 1px solid rgba(108, 92, 231, 0.3);
    }
    
    /* ---- Headers ---- */
    h1 {
        background: linear-gradient(90deg, #6c5ce7, #a29bfe, #fd79a8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #dfe6e9 !important;
        font-weight: 600 !important;
    }
    
    /* ---- Metric cards ---- */
    [data-testid="stMetric"] {
        background: rgba(108, 92, 231, 0.1);
        border: 1px solid rgba(108, 92, 231, 0.25);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }
    [data-testid="stMetricLabel"] {
        color: #a29bfe !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricValue"] * {
        overflow: visible !important;
        text-overflow: unset !important;
        white-space: normal !important;
    }
    
    /* ---- Custom date-range card ---- */
    .date-range-card {
        background: rgba(108, 92, 231, 0.1);
        border: 1px solid rgba(108, 92, 231, 0.25);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
        height: 100%;
    }
    .date-range-card .label {
        color: #a29bfe;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
        font-weight: 400;
    }
    .date-range-card .value {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.3;
    }
    
    /* ---- Cards / containers ---- */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
    }
    
    /* ---- Anomaly alert ---- */
    .anomaly-card {
        background: rgba(214, 48, 49, 0.08);
        border-left: 4px solid #e17055;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
        color: #dfe6e9;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .anomaly-card strong {
        color: #fab1a0;
    }
    
    /* ---- Success card ---- */
    .success-card {
        background: rgba(0, 206, 201, 0.08);
        border-left: 4px solid #00cec9;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
        color: #dfe6e9;
    }

    /* ---- Insight card ---- */
    .insight-card {
        background: rgba(108, 92, 231, 0.08);
        border-left: 4px solid #6c5ce7;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
        color: #dfe6e9;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .insight-card strong {
        color: #a29bfe;
    }

    /* ---- Executive summary ---- */
    .exec-summary {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(108, 92, 231, 0.15);
        border-radius: 16px;
        padding: 24px 28px;
        margin: 16px 0 24px 0;
        color: #dfe6e9;
        font-size: 0.95rem;
        line-height: 1.8;
    }
    .exec-summary h4 {
        color: #a29bfe !important;
        margin-bottom: 12px;
        font-size: 1rem;
    }
    .exec-summary .highlight {
        color: #00cec9;
        font-weight: 600;
    }
    .exec-summary .warn {
        color: #e17055;
        font-weight: 600;
    }
    
    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(108, 92, 231, 0.1);
        border-radius: 8px 8px 0 0;
        border: 1px solid rgba(108, 92, 231, 0.2);
        color: #a29bfe;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(108, 92, 231, 0.25) !important;
        border-bottom-color: transparent !important;
        color: #ffffff !important;
    }
    
    /* ---- Slider ---- */
    .stSlider > div > div > div > div {
        background: #6c5ce7 !important;
    }
    
    /* ---- Info text ---- */
    .info-text {
        color: #b2bec3;
        font-size: 0.88rem;
        line-height: 1.7;
    }
    
    /* ---- Divider ---- */
    hr {
        border-color: rgba(108, 92, 231, 0.2) !important;
    }

    /* ---- Grade badge ---- */
    .grade-badge {
        display: inline-block;
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        padding: 20px 30px;
        border-radius: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.markdown("## ⚡ NatWest Advanced Forecasting")
    st.markdown("<p class='info-text'>Upload your data and configure the forecasting engine.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ---- Data source ----
    st.markdown("### 📂 Data Source")
    data_source = st.radio(
        "Choose data source:",
        ["Use sample dataset", "Upload CSV"],
        label_visibility="collapsed",
    )

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            help="CSV must have columns: `date` and a numeric metric column",
        )
        metric_col = st.text_input("Metric column name", value="traffic", help="Name of the numeric column to forecast")
    else:
        metric_col = "traffic"

    st.markdown("---")

    # ---- Forecast settings ----
    st.markdown("### ⚙️ Forecast Settings")
    weeks_ahead = st.slider("Forecast horizon (weeks)", min_value=1, max_value=6, value=4, help="How many weeks to project forward")
    z_threshold = st.slider("Anomaly sensitivity (Z-score)", min_value=1.5, max_value=4.0, value=2.5, step=0.1, help="Lower = more sensitive (more anomalies flagged)")

    st.markdown("---")

    # ---- Scenario settings ----
    st.markdown("### 🔮 What-If Scenario")
    scenario_pct = st.slider("Volume change (%)", min_value=-50, max_value=100, value=15, step=5, help="Simulate a percentage change in the metric")

    st.markdown("---")
    st.markdown("<p class='info-text'>Built for NatWest<br/>'Code for Purpose' Hackathon</p>", unsafe_allow_html=True)


# =====================================================================
# Load data
# =====================================================================
@st.cache_data
def load_sample_data() -> pd.DataFrame:
    """Load the bundled sample dataset from assets/."""
    sample_path = os.path.join(PROJECT_ROOT, "assets", "sample_data.csv")
    return pd.read_csv(sample_path)


def load_data() -> pd.DataFrame:
    """Load data from uploaded file or sample dataset."""
    if data_source == "Upload CSV" and uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return load_sample_data()


df = load_data()

# Validate data
if "date" not in df.columns:
    st.error("❌ CSV must contain a `date` column.")
    st.stop()
if metric_col not in df.columns:
    st.error(f"❌ CSV must contain a `{metric_col}` column.")
    st.stop()

df["date"] = pd.to_datetime(df["date"])

# =====================================================================
# Pre-compute all engines (used across tabs & summary)
# =====================================================================
forecaster = Forecaster(df, target_col=metric_col, date_col="date")
forecast_df = forecaster.generate_forecast(weeks_ahead=weeks_ahead)

detector = AnomalyDetector(df, target_col=metric_col, date_col="date")
anomalies = detector.detect_anomalies(window_size=4, dynamic_z_score_threshold=z_threshold)

analyzer = TrendAnalyzer(df, target_col=metric_col, date_col="date")
trend_stats = analyzer.compute_summary_stats()

backtester = Backtester(df, target_col=metric_col, date_col="date")
backtest_results = backtester.run_backtest(holdout_weeks=weeks_ahead)

# =====================================================================
# Header
# =====================================================================
st.markdown("# Advanced Predictive Forecasting Dashboard")
st.markdown("<p class='info-text'>Transparent, explainable forecasting — powered by Holt-Winters Exponential Smoothing with walk-forward validation, trend decomposition, and multi-model comparison.</p>", unsafe_allow_html=True)

# ---- KPI row ----
date_min = df['date'].min().strftime("%b %d, '%y")
date_max = df['date'].max().strftime("%b %d, '%y")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Data Points", f"{len(df)}")
with col2:
    st.markdown(f"""
    <div class="date-range-card">
        <div class="label">Date Range</div>
        <div class="value">{date_min}<br/>→ {date_max}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.metric("Latest Value", f"{df[metric_col].iloc[-1]:.0f}")
with col4:
    trend = df[metric_col].iloc[-1] - df[metric_col].iloc[-5] if len(df) >= 5 else 0
    st.metric("5-Week Trend", f"{trend:+.0f}")


# =====================================================================
# Auto-Generated Executive Summary
# =====================================================================
forecast_direction = "increase" if forecast_df["Likely_Estimate"].iloc[-1] > df[metric_col].iloc[-1] else "decrease"
forecast_change_pct = abs((forecast_df["Likely_Estimate"].iloc[-1] - df[metric_col].iloc[-1]) / df[metric_col].iloc[-1]) * 100
anomaly_count = len(anomalies)

mape = backtest_results['metrics']['mape']
grade, grade_color = Backtester.get_accuracy_grade(mape)
mape_str = f"{mape:.1f}%" if mape is not None else "N/A"

st.markdown(f"""
<div class="exec-summary">
    <h4>📋 Executive Summary — Auto-Generated Insights</h4>
    <p>
        Based on <span class="highlight">{len(df)} weeks</span> of historical data, our Holt-Winters model projects
        a <span class="{'highlight' if forecast_direction == 'increase' else 'warn'}">{forecast_change_pct:.1f}% {forecast_direction}</span>
        over the next <span class="highlight">{weeks_ahead} week(s)</span>.
        The model's backtested accuracy is <span class="highlight">MAPE {mape_str} (Grade {grade})</span>,
        meaning predictions are historically within ~{mape_str} of actuals.
    </p>
    <p>
        {"<span class='warn'>⚠️ " + str(anomaly_count) + " anomalies</span> were detected in the dataset — review the Spot Trouble tab for AI-powered explanations." if anomaly_count > 0 else "<span class='highlight'>✅ No anomalies detected</span> — the data appears clean and stable."}
        The overall trend is <span class="highlight">{trend_stats['trend_direction']}</span>
        with a weekly volatility of <span class="highlight">{trend_stats['volatility_pct']:.1f}%</span>.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================================================================
# Tabs
# =====================================================================
tab_forecast, tab_anomalies, tab_scenario, tab_backtest, tab_decompose, tab_goal, tab_health = st.tabs([
    "📈 Forecast", "🔍 Spot Trouble", "🔮 What-If Scenario",
    "🎯 Model Accuracy", "📊 Trend Decomposition",
    "🏁 Goal Tracking", "🩺 Data Health Check"
])

# =====================================================================
# TAB 1: Forecast
# =====================================================================
with tab_forecast:
    st.markdown("### Short-Term Forecast")
    st.markdown(f"<p class='info-text'>Projecting <strong>{weeks_ahead} week(s)</strong> ahead using Holt-Winters vs. a 4-week moving average baseline.</p>", unsafe_allow_html=True)

    # ---- Build combined chart data ----
    hist_chart = df[["date", metric_col]].copy()
    hist_chart.columns = ["Date", "Historical"]

    forecast_chart = forecast_df[["Date", "Likely_Estimate", "Baseline_Avg", "Low_Bound", "High_Bound"]].copy()
    forecast_chart.columns = ["Date", "Likely Estimate", "Baseline Avg", "Low Bound", "High Bound"]

    # Combine for a continuous line: add last historical point to forecast
    bridge_row = pd.DataFrame({
        "Date": [hist_chart["Date"].iloc[-1]],
        "Likely Estimate": [hist_chart["Historical"].iloc[-1]],
        "Baseline Avg": [hist_chart["Historical"].iloc[-1]],
        "Low Bound": [hist_chart["Historical"].iloc[-1]],
        "High Bound": [hist_chart["Historical"].iloc[-1]],
    })
    forecast_chart = pd.concat([bridge_row, forecast_chart], ignore_index=True)

    # ---- Chart ----
    # Historical line
    hist_line = alt.Chart(hist_chart).mark_line(
        strokeWidth=2.5, color="#6c5ce7"
    ).encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Historical:Q", title=metric_col.capitalize(), scale=alt.Scale(zero=False)),
        tooltip=["Date:T", "Historical:Q"],
    )

    hist_points = alt.Chart(hist_chart).mark_circle(
        size=30, color="#a29bfe", opacity=0.6
    ).encode(x="Date:T", y="Historical:Q")

    # Forecast line
    forecast_line = alt.Chart(forecast_chart).mark_line(
        strokeWidth=2.5, color="#00cec9", strokeDash=[6, 4]
    ).encode(
        x="Date:T",
        y=alt.Y("Likely Estimate:Q", scale=alt.Scale(zero=False)),
        tooltip=["Date:T", "Likely Estimate:Q"],
    )

    # Baseline
    baseline_line = alt.Chart(forecast_chart).mark_line(
        strokeWidth=1.5, color="#fdcb6e", strokeDash=[4, 4], opacity=0.7
    ).encode(
        x="Date:T",
        y="Baseline Avg:Q",
        tooltip=["Date:T", "Baseline Avg:Q"],
    )

    # Confidence band
    band = alt.Chart(forecast_chart).mark_area(
        opacity=0.15, color="#00cec9"
    ).encode(
        x="Date:T",
        y="Low Bound:Q",
        y2="High Bound:Q",
    )

    chart = (hist_line + hist_points + band + forecast_line + baseline_line).properties(
        height=420,
    ).configure_axis(
        labelColor="#b2bec3",
        titleColor="#dfe6e9",
        gridColor="rgba(255,255,255,0.05)",
    ).configure_view(
        strokeWidth=0,
    )

    st.altair_chart(chart, use_container_width=True)

    # ---- Legend ----
    legend_cols = st.columns(4)
    legend_cols[0].markdown("🟣 **Historical Data**")
    legend_cols[1].markdown("🟢 **Likely Estimate**")
    legend_cols[2].markdown("🟡 **Baseline (Moving Avg)**")
    legend_cols[3].markdown("🔵 **90% Confidence Band**")

    # ---- Data table ----
    with st.expander("View raw forecast data"):
        st.dataframe(forecast_df.round(2), use_container_width=True, hide_index=True)

    # ---- Download Button ----
    st.markdown("#### 📥 Export Results")
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Forecast CSV",
        data=csv,
        file_name='forecast_results.csv',
        mime='text/csv',
    )


# =====================================================================
# TAB 2: Anomaly Detection
# =====================================================================
with tab_anomalies:
    st.markdown("### Spot Trouble — Anomaly Detection")
    st.markdown(f"<p class='info-text'>Scanning historical data with a Z-score threshold of <strong>{z_threshold}</strong>. Points deviating beyond this threshold from the rolling mean are flagged.</p>", unsafe_allow_html=True)

    if anomalies.empty:
        st.markdown('<div class="success-card">No significant anomalies detected. The data looks healthy!</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"**{len(anomalies)} anomaly(ies) detected:**")
        explainer = AnomalyExplainer()

        for date_idx, row in anomalies.iterrows():
            date_str = str(date_idx.date()) if hasattr(date_idx, "date") else str(date_idx)
            actual = row[metric_col]
            expected = row["Rolling_Mean"]
            z_score = row["Z_Score"]
            direction = "Spike" if actual > expected else "Dip"
            diff_pct = abs((actual - expected) / expected) * 100

            explanation = explainer.generate_explanation(
                date=date_str,
                actual_value=actual,
                expected_value=expected,
                z_score=z_score,
            )

            st.markdown(f"""
            <div class="anomaly-card">
                <strong>⚠️ {direction} on {date_str}</strong><br/>
                Value: <strong>{actual:.0f}</strong> &nbsp;|&nbsp; Expected: <strong>{expected:.1f}</strong> &nbsp;|&nbsp;
                Deviation: <strong>{diff_pct:.0f}%</strong> &nbsp;|&nbsp; Z-Score: <strong>{z_score:.2f}</strong>
                <br/><br/>
                {explanation}
            </div>
            """, unsafe_allow_html=True)

        # Anomaly scatter overlay chart
        st.markdown("#### Anomaly Overlay")
        hist_for_anomaly = df[["date", metric_col]].copy()
        hist_for_anomaly.columns = ["Date", "Value"]

        anomaly_points = []
        for date_idx, row in anomalies.iterrows():
            anomaly_points.append({
                "Date": date_idx,
                "Value": row[metric_col],
                "Type": "Spike" if row[metric_col] > row["Rolling_Mean"] else "Dip",
            })
        anomaly_df = pd.DataFrame(anomaly_points)

        base_line = alt.Chart(hist_for_anomaly).mark_line(
            strokeWidth=2, color="#6c5ce7"
        ).encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Value:Q", title=metric_col.capitalize(), scale=alt.Scale(zero=False)),
        )

        anomaly_scatter = alt.Chart(anomaly_df).mark_point(
            size=200, filled=True, opacity=0.9
        ).encode(
            x="Date:T",
            y="Value:Q",
            color=alt.Color("Type:N", scale=alt.Scale(
                domain=["Spike", "Dip"],
                range=["#e17055", "#00b894"]
            )),
            tooltip=["Date:T", "Value:Q", "Type:N"],
        )

        anomaly_chart = (base_line + anomaly_scatter).properties(height=350).configure_axis(
            labelColor="#b2bec3", titleColor="#dfe6e9", gridColor="rgba(255,255,255,0.05)"
        ).configure_view(strokeWidth=0)

        st.altair_chart(anomaly_chart, use_container_width=True)


# =====================================================================
# TAB 3: Scenario Forecasting
# =====================================================================
with tab_scenario:
    st.markdown("### What-If Scenario Modelling")
    st.markdown(f"<p class='info-text'>Comparing the baseline forecast against a <strong>{scenario_pct:+d}%</strong> volume adjustment. Move the slider in the sidebar to update in real-time.</p>", unsafe_allow_html=True)

    scenario_df = ScenarioForecaster.apply_scenario(forecast_df, percentage_change=float(scenario_pct))
    scenario_col = f"Scenario_({float(scenario_pct):+.1f}%)"

    # ---- KPI impact ----
    total_impact = scenario_df["Numerical_Impact"].sum()
    avg_impact = scenario_df["Numerical_Impact"].mean()

    impact_cols = st.columns(3)
    impact_cols[0].metric("Scenario", f"{scenario_pct:+d}%")
    impact_cols[1].metric("Total Impact", f"{total_impact:+.1f}")
    impact_cols[2].metric("Avg Weekly Impact", f"{avg_impact:+.1f}")

    # ---- Side-by-side chart ----
    scenario_chart_data = scenario_df[["Date", "Likely_Estimate", scenario_col]].copy()
    scenario_chart_data.columns = ["Date", "Baseline Forecast", "Scenario Forecast"]

    scenario_melted = scenario_chart_data.melt(
        id_vars=["Date"],
        value_vars=["Baseline Forecast", "Scenario Forecast"],
        var_name="Type",
        value_name="Value",
    )

    scenario_lines = alt.Chart(scenario_melted).mark_line(
        strokeWidth=3, point=alt.OverlayMarkDef(size=60)
    ).encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Value:Q", title=metric_col.capitalize(), scale=alt.Scale(zero=False)),
        color=alt.Color("Type:N", scale=alt.Scale(
            domain=["Baseline Forecast", "Scenario Forecast"],
            range=["#6c5ce7", "#e17055"]
        )),
        tooltip=["Date:T", "Type:N", "Value:Q"],
    )

    combined = alt.layer(
        scenario_lines
    ).properties(height=380).configure_axis(
        labelColor="#b2bec3", titleColor="#dfe6e9", gridColor="rgba(255,255,255,0.05)"
    ).configure_view(strokeWidth=0)

    st.altair_chart(combined, use_container_width=True)

    # ---- Data table ----
    with st.expander("View scenario comparison data"):
        display_df = scenario_df.copy()
        display_df.columns = ["Date", "Baseline Estimate", f"Scenario ({scenario_pct:+d}%)", "Impact"]
        st.dataframe(display_df.round(2), use_container_width=True, hide_index=True)

    # ---- Insight box ----
    if scenario_pct > 0:
        st.markdown(f"""
        <div class="success-card">
            <strong>💡 Insight:</strong> A {scenario_pct:+d}% increase in volume would add approximately
            <strong>{avg_impact:+.1f} units/week</strong> to the projected metric, totalling
            <strong>{total_impact:+.1f}</strong> additional units over the {weeks_ahead}-week forecast window.
        </div>
        """, unsafe_allow_html=True)
    elif scenario_pct < 0:
        st.markdown(f"""
        <div class="anomaly-card">
            <strong>⚠️ Risk Alert:</strong> A {scenario_pct:+d}% reduction would decrease the metric by approximately
            <strong>{abs(avg_impact):.1f} units/week</strong>, representing a total loss of
            <strong>{abs(total_impact):.1f}</strong> units over the {weeks_ahead}-week window.
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# TAB 4: Model Accuracy (Backtesting)
# =====================================================================
with tab_backtest:
    st.markdown("### 🎯 Model Accuracy — Walk-Forward Backtesting")
    st.markdown("<p class='info-text'>The gold standard for time-series validation. We retrain the model at each historical checkpoint and compare predictions against actuals to measure real-world reliability.</p>", unsafe_allow_html=True)

    metrics = backtest_results['metrics']
    predictions = backtest_results['predictions']

    if predictions.empty:
        st.warning("Not enough data points to perform backtesting. Upload a larger dataset.")
    else:
        # ---- Grade & Metrics Row ----
        g_col1, g_col2, g_col3, g_col4, g_col5 = st.columns([1, 1, 1, 1, 1])

        with g_col1:
            st.markdown(f"""
            <div style="text-align:center; padding: 20px; background: rgba(255,255,255,0.04); border-radius: 16px; border: 2px solid {grade_color};">
                <div class="grade-badge" style="color: {grade_color};">{grade}</div>
                <p style="color: #b2bec3; margin-top: 8px; font-size: 0.85rem;">Model Grade</p>
            </div>
            """, unsafe_allow_html=True)

        with g_col2:
            st.metric("MAPE", mape_str)
        with g_col3:
            st.metric("MAE", f"{metrics['mae']:.2f}" if metrics['mae'] else "N/A")
        with g_col4:
            st.metric("RMSE", f"{metrics['rmse']:.2f}" if metrics['rmse'] else "N/A")
        with g_col5:
            da = metrics['directional_accuracy']
            st.metric("Direction Accuracy", f"{da:.0f}%" if da else "N/A")

        # ---- Interpretation ----
        st.markdown(f"""
        <div class="insight-card">
            <strong>📝 What this means:</strong> {Backtester.interpret_mape(metrics['mape'])} <br/>
            <strong>MAPE</strong> = Mean Absolute Percentage Error — on average, predictions deviate {mape_str} from actual values.<br/>
            <strong>Direction Accuracy</strong> = How often the model correctly predicts whether the metric goes up or down.
        </div>
        """, unsafe_allow_html=True)

        # ---- Actual vs Predicted chart ----
        st.markdown("#### Actual vs. Predicted (Backtest)")

        bt_melted = predictions[["Date", "Actual", "Predicted"]].melt(
            id_vars=["Date"],
            value_vars=["Actual", "Predicted"],
            var_name="Type",
            value_name="Value",
        )

        bt_chart = alt.Chart(bt_melted).mark_line(
            strokeWidth=2.5, point=alt.OverlayMarkDef(size=40)
        ).encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Value:Q", title=metric_col.capitalize(), scale=alt.Scale(zero=False)),
            color=alt.Color("Type:N", scale=alt.Scale(
                domain=["Actual", "Predicted"],
                range=["#6c5ce7", "#00cec9"]
            )),
            tooltip=["Date:T", "Type:N", "Value:Q"],
        ).properties(height=380).configure_axis(
            labelColor="#b2bec3", titleColor="#dfe6e9", gridColor="rgba(255,255,255,0.05)"
        ).configure_view(strokeWidth=0)

        st.altair_chart(bt_chart, use_container_width=True)

        # ---- Error distribution chart ----
        st.markdown("#### Prediction Error Distribution")

        error_hist_data = predictions[["Abs_Pct_Error"]].copy()
        error_hist_data.columns = ["Absolute % Error"]

        error_chart = alt.Chart(error_hist_data).mark_bar(
            color="#6c5ce7", opacity=0.7, cornerRadiusTopLeft=4, cornerRadiusTopRight=4
        ).encode(
            x=alt.X("Absolute % Error:Q", bin=alt.Bin(maxbins=15), title="Absolute % Error"),
            y=alt.Y("count()", title="Frequency"),
        ).properties(height=250).configure_axis(
            labelColor="#b2bec3", titleColor="#dfe6e9", gridColor="rgba(255,255,255,0.05)"
        ).configure_view(strokeWidth=0)

        st.altair_chart(error_chart, use_container_width=True)

        # ---- Data table ----
        with st.expander("View detailed backtest results"):
            st.dataframe(predictions.round(2), use_container_width=True, hide_index=True)


# =====================================================================
# TAB 5: Trend Decomposition
# =====================================================================
with tab_decompose:
    st.markdown("### 📊 Trend Decomposition & Autocorrelation")
    st.markdown("<p class='info-text'>Breaking the time series into its fundamental components — <strong>Trend</strong> (long-term direction), <strong>Seasonal</strong> (repeating cycles), and <strong>Residual</strong> (noise). This reveals the hidden structure driving your metric.</p>", unsafe_allow_html=True)

    decomp_period = st.slider("Seasonal period", min_value=2, max_value=min(26, len(df) // 2), value=min(4, len(df) // 4), help="Number of periods per seasonal cycle")

    decomp = analyzer.decompose(period=decomp_period)

    # ---- Observed + Trend chart ----
    st.markdown("#### Observed Data with Trend Line")

    obs_df = pd.DataFrame({
        "Date": decomp['observed'].index,
        "Observed": decomp['observed'].values,
    })
    trend_df = pd.DataFrame({
        "Date": decomp['trend'].index,
        "Trend": decomp['trend'].values,
    }).dropna()

    obs_line = alt.Chart(obs_df).mark_line(strokeWidth=1.5, color="#6c5ce7", opacity=0.6).encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Observed:Q", title=metric_col.capitalize(), scale=alt.Scale(zero=False)),
    )
    trend_line = alt.Chart(trend_df).mark_line(strokeWidth=3, color="#00cec9").encode(
        x="Date:T",
        y=alt.Y("Trend:Q", scale=alt.Scale(zero=False)),
    )

    obs_trend_chart = (obs_line + trend_line).properties(height=300).configure_axis(
        labelColor="#b2bec3", titleColor="#dfe6e9", gridColor="rgba(255,255,255,0.05)"
    ).configure_view(strokeWidth=0)

    st.altair_chart(obs_trend_chart, use_container_width=True)

    # ---- Seasonal & Residual side-by-side ----
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Seasonal Component")
        seasonal_df = pd.DataFrame({
            "Date": decomp['seasonal'].index,
            "Seasonal": decomp['seasonal'].values,
        })
        seasonal_chart = alt.Chart(seasonal_df).mark_area(
            color="#fdcb6e", opacity=0.5, line={"color": "#fdcb6e", "strokeWidth": 2}
        ).encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Seasonal:Q", title="Seasonal Effect"),
        ).properties(height=250).configure_axis(
            labelColor="#b2bec3", titleColor="#dfe6e9", gridColor="rgba(255,255,255,0.05)"
        ).configure_view(strokeWidth=0)
        st.altair_chart(seasonal_chart, use_container_width=True)

    with c2:
        st.markdown("#### Residual (Noise)")
        resid_df = pd.DataFrame({
            "Date": decomp['residual'].index,
            "Residual": decomp['residual'].values,
        }).dropna()
        resid_chart = alt.Chart(resid_df).mark_bar(
            color="#e17055", opacity=0.6, cornerRadiusTopLeft=2, cornerRadiusTopRight=2
        ).encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Residual:Q", title="Residual"),
        ).properties(height=250).configure_axis(
            labelColor="#b2bec3", titleColor="#dfe6e9", gridColor="rgba(255,255,255,0.05)"
        ).configure_view(strokeWidth=0)
        st.altair_chart(resid_chart, use_container_width=True)

    # ---- Autocorrelation ----
    st.markdown("#### Autocorrelation Function (ACF)")
    st.markdown("<p class='info-text'>Shows how strongly the metric correlates with its own past values. Bars exceeding the dashed significance line indicate meaningful lag relationships.</p>", unsafe_allow_html=True)

    acf_df = analyzer.compute_acf(nlags=min(20, len(df) // 2 - 1))

    acf_bars = alt.Chart(acf_df).mark_bar(
        color="#6c5ce7", opacity=0.8, cornerRadiusTopLeft=3, cornerRadiusTopRight=3
    ).encode(
        x=alt.X("Lag:O", title="Lag"),
        y=alt.Y("ACF:Q", title="Autocorrelation"),
    )

    sig_upper = alt.Chart(acf_df).mark_rule(
        color="#e17055", strokeDash=[4, 4], opacity=0.7
    ).encode(y="Significance_Upper:Q")

    sig_lower = alt.Chart(acf_df).mark_rule(
        color="#e17055", strokeDash=[4, 4], opacity=0.7
    ).encode(y="Significance_Lower:Q")

    acf_chart = (acf_bars + sig_upper + sig_lower).properties(height=280).configure_axis(
        labelColor="#b2bec3", titleColor="#dfe6e9", gridColor="rgba(255,255,255,0.05)"
    ).configure_view(strokeWidth=0)

    st.altair_chart(acf_chart, use_container_width=True)

    # ---- Key Statistics ----
    st.markdown("#### Key Trend Statistics")
    stat_cols = st.columns(4)
    stat_cols[0].metric("Trend Direction", trend_stats['trend_direction'])
    stat_cols[1].metric("Weekly Volatility", f"{trend_stats['volatility_pct']:.1f}%")
    stat_cols[2].metric("Trend Slope", f"{trend_stats['trend_slope']:+.2f}")
    stat_cols[3].metric("Range", f"{trend_stats['min']:.0f} – {trend_stats['max']:.0f}")


# =====================================================================
# TAB 6: Goal Tracking
# =====================================================================
with tab_goal:
    st.markdown("### 🏁 Goal Tracking & Probability of Success")
    st.markdown("<p class='info-text'>Set a target value for a specific future week, and the system will calculate the statistical probability of achieving it based on historical variance.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        target_dates = forecast_df["Date"].dt.strftime("%Y-%m-%d").tolist()
        selected_date_str = st.selectbox("Select Target Week", target_dates)

        # Get forecast for this date
        target_row = forecast_df[forecast_df["Date"].dt.strftime("%Y-%m-%d") == selected_date_str].iloc[0]
        likely_est = target_row["Likely_Estimate"]
        high_bound = target_row["High_Bound"]

        # Calculate Standard Error based on our 90% confidence interval logic in forecaster.py
        # Margin of Error (ME) = 1.645 * std_err  => std_err = ME / 1.645
        margin_of_error = high_bound - likely_est
        std_err = margin_of_error / 1.645 if margin_of_error > 0 else 1.0

        target_value = st.number_input("Target Value", value=float(likely_est * 1.1), step=10.0, help="The business goal you want to reach by the selected week.")

    with col2:
        # Calculate Z and probability
        z_val = (target_value - likely_est) / std_err
        # sf = 1 - cdf (survival function). Probability of value being >= target_value
        prob_success = norm.sf(z_val) * 100

        st.markdown("#### Probability of Reaching Target")

        if prob_success >= 75:
            p_color = "#00b894"
            p_msg = "✅ Highly Likely"
        elif prob_success >= 40:
            p_color = "#fdcb6e"
            p_msg = "⚡ Possible, but challenging"
        else:
            p_color = "#d63031"
            p_msg = "⚠️ Unlikely (Requires intervention)"

        st.markdown(f"""
        <div style="padding: 20px; background: rgba(0,0,0,0.2); border-radius: 12px; border-left: 5px solid {p_color};">
            <h2 style="margin: 0; color: {p_color} !important;">{prob_success:.1f}%</h2>
            <p style="margin: 5px 0 0 0; color: #dfe6e9;">{p_msg}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(f"""
        **Context:** Our model predicts a central estimate of **{likely_est:.1f}** for {selected_date_str}. 
        To reach your goal of **{target_value:.1f}**, you need a **{((target_value / likely_est) - 1) * 100:+.1f}%** difference from the organic baseline trajectory.
        """)

    # ---- What-if bridge ----
    st.markdown("---")
    st.markdown("#### Reverse Scenario: What growth rate do you need?")

    required_growth_pct = ((target_value / likely_est) - 1) * 100
    st.markdown(f"""
    <div class="insight-card">
        <strong>🔗 Linking to Scenario Tab:</strong> To reach your target of <strong>{target_value:.1f}</strong>,
        you would need a <strong>{required_growth_pct:+.1f}%</strong> uplift over the organic forecast.
        Try setting the What-If slider in the sidebar to <strong>{int(round(required_growth_pct))}%</strong> to see the full scenario analysis.
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# TAB 7: Data Health Check
# =====================================================================
with tab_health:
    st.markdown("### 🩺 Data Quality Health Check")
    st.markdown("<p class='info-text'>Automatically scanning the uploaded dataset for missing values, gaps, and extreme outliers. Good data quality is crucial for accurate forecasting.</p>", unsafe_allow_html=True)

    health_score = 100
    health_issues = []

    # 1. Missing Values
    null_count = df[metric_col].isnull().sum()
    if null_count > 0:
        health_score -= 20
        health_issues.append(f"Found {null_count} missing (NaN) values in the metric column.")

    # 2. Date Gaps
    date_diffs = df['date'].diff().dt.days
    expected_freq = date_diffs.mode().iloc[0] if not date_diffs.dropna().empty else 7
    anomalous_gaps = date_diffs[date_diffs > 1.5 * expected_freq].count()
    if anomalous_gaps > 0:
        health_score -= min(anomalous_gaps * 5, 30)
        health_issues.append(f"Detected {anomalous_gaps} irregular gaps in the timeline. The data frequency should ideally be consistent ({expected_freq:.0f} days).")

    # 3. Extreme Outliers (Z > 3.5)
    mean_val = df[metric_col].mean()
    std_val = df[metric_col].std()
    extreme_outliers = df[abs(df[metric_col] - mean_val) > 3.5 * std_val]
    if not extreme_outliers.empty:
        health_score -= min(len(extreme_outliers) * 5, 20)
        health_issues.append(f"Found {len(extreme_outliers)} extreme outliers (Z > 3.5). These can skew the Holt-Winters initialization.")

    # 4. Data size
    if len(df) < 12:
        health_score -= 15
        health_issues.append(f"Only {len(df)} data points. At least 12 weekly observations are recommended for reliable Holt-Winters fitting.")

    # 5. Duplicate dates
    dup_dates = df['date'].duplicated().sum()
    if dup_dates > 0:
        health_score -= 10
        health_issues.append(f"Found {dup_dates} duplicate date entries. Each date should appear exactly once.")

    health_score = max(0, health_score)

    colA, colB = st.columns([1, 2])
    with colA:
        if health_score >= 90:
            color = "#00b894"
            status = "Excellent"
        elif health_score >= 70:
            color = "#fdcb6e"
            status = "Fair"
        else:
            color = "#d63031"
            status = "Needs Attention"

        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background: rgba(255,255,255,0.05); border-radius: 12px; border: 2px solid {color};">
            <h1 style="font-size: 4rem; color: {color} !important; border-bottom: none; margin: 0; padding: 0;">{int(health_score)}%</h1>
            <h3 style="color: {color} !important; margin: 10px 0 0 0;">{status}</h3>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown("#### Diagnostic Report")
        if not health_issues:
            st.markdown('<div class="success-card">✅ Your dataset is perfectly structured and contains no structural warnings!</div>', unsafe_allow_html=True)
        else:
            for issue in health_issues:
                st.markdown(f'<div class="anomaly-card">⚠️ {issue}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Metric Summary Statistics:**")
        stats_df = pd.DataFrame(df[metric_col].describe()).T
        st.dataframe(stats_df.round(2), use_container_width=True)


# =====================================================================
# Footer
# =====================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#636e72; font-size:0.8rem;'>"
    "Built with ❤️ for NatWest 'Code for Purpose' Hackathon &nbsp;|&nbsp; "
    "Powered by Holt-Winters Exponential Smoothing &nbsp;|&nbsp; "
    "Walk-Forward Backtesting &nbsp;|&nbsp; "
    "Trend Decomposition Engine &nbsp;|&nbsp; "
    "Automated Insights Generation"
    "</p>",
    unsafe_allow_html=True,
)
