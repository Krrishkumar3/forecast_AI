"""
dashboard.py — Interactive Streamlit Dashboard for NatWest Forecasting.

A single-page application that lets users:
  1. Upload a CSV or use the bundled sample dataset
  2. Select a forecast horizon (1–6 weeks) and view line charts with
     historical data, baseline, likely estimate, and shaded uncertainty bounds
  3. Browse detected anomalies with AI-generated explanations
  4. Adjust a "What-if" slider and see scenario projections update in real-time

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

from src.core.forecaster import Forecaster
from src.core.anomaly_detector import AnomalyDetector
from src.core.scenario_runner import ScenarioForecaster
from src.core.explainer import AnomalyExplainer


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
        font-size: clamp(1rem, 2vw, 1.75rem) !important;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
        white-space: normal !important;
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
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.markdown("## NatWest Advanced Forecasting")
    st.markdown("<p class='info-text'>Upload your data and configure the forecasting engine.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ---- Data source ----
    st.markdown("### Data Source")
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
    st.markdown("### Forecast Settings")
    weeks_ahead = st.slider("Forecast horizon (weeks)", min_value=1, max_value=6, value=4, help="How many weeks to project forward")
    z_threshold = st.slider("Anomaly sensitivity (Z-score)", min_value=1.5, max_value=4.0, value=2.5, step=0.1, help="Lower = more sensitive (more anomalies flagged)")

    st.markdown("---")

    # ---- Scenario settings ----
    st.markdown("### What-If Scenario")
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
# Header
# =====================================================================
st.markdown("# Advanced Predictive Forecasting Dashboard")
st.markdown("<p class='info-text'>Transparent, explainable forecasting — powered by Holt-Winters Exponential Smoothing with simple baseline comparison.</p>", unsafe_allow_html=True)

# ---- KPI row ----
date_min = df['date'].min().strftime("%b %d, '%y")
date_max = df['date'].max().strftime("%b %d, '%y")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Data Points", f"{len(df)}")
with col2:
    st.metric("Date Range", f"{date_min} → {date_max}")
with col3:
    st.metric("Latest Value", f"{df[metric_col].iloc[-1]:.0f}")
with col4:
    trend = df[metric_col].iloc[-1] - df[metric_col].iloc[-5] if len(df) >= 5 else 0
    st.metric("5-Week Trend", f"{trend:+.0f}")

st.markdown("---")

# =====================================================================
# Tabs
# =====================================================================
tab_forecast, tab_anomalies, tab_scenario = st.tabs([
    "Forecast", "Spot Trouble", "What-If Scenario"
])

# =====================================================================
# TAB 1: Forecast
# =====================================================================
with tab_forecast:
    st.markdown("### Short-Term Forecast")
    st.markdown(f"<p class='info-text'>Projecting <strong>{weeks_ahead} week(s)</strong> ahead using Holt-Winters vs. a 4-week moving average baseline.</p>", unsafe_allow_html=True)

    # Generate forecast
    forecaster = Forecaster(df, target_col=metric_col, date_col="date")
    forecast_df = forecaster.generate_forecast(weeks_ahead=weeks_ahead)

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
    import altair as alt

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
    legend_cols[0].markdown("**Historical Data**")
    legend_cols[1].markdown("**Likely Estimate**")
    legend_cols[2].markdown("**Baseline (Moving Avg)**")
    legend_cols[3].markdown("**90% Confidence Band**")

    # ---- Data table ----
    with st.expander("View raw forecast data"):
        st.dataframe(forecast_df.round(2), use_container_width=True, hide_index=True)


# =====================================================================
# TAB 2: Anomaly Detection
# =====================================================================
with tab_anomalies:
    st.markdown("### Spot Trouble — Anomaly Detection")
    st.markdown(f"<p class='info-text'>Scanning historical data with a Z-score threshold of <strong>{z_threshold}</strong>. Points deviating beyond this threshold from the rolling mean are flagged.</p>", unsafe_allow_html=True)

    detector = AnomalyDetector(df, target_col=metric_col, date_col="date")
    anomalies = detector.detect_anomalies(window_size=4, dynamic_z_score_threshold=z_threshold)

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
                <strong>{direction} on {date_str}</strong><br/>
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

    # Impact bars
    impact_bars = alt.Chart(scenario_df).mark_bar(
        opacity=0.25, color="#fab1a0" if scenario_pct >= 0 else "#00b894"
    ).encode(
        x="Date:T",
        y=alt.Y("Numerical_Impact:Q", title="Impact"),
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
            <strong>Insight:</strong> A {scenario_pct:+d}% increase in volume would add approximately
            <strong>{avg_impact:+.1f} units/week</strong> to the projected metric, totalling
            <strong>{total_impact:+.1f}</strong> additional units over the {weeks_ahead}-week forecast window.
        </div>
        """, unsafe_allow_html=True)
    elif scenario_pct < 0:
        st.markdown(f"""
        <div class="anomaly-card">
            <strong>Risk Alert:</strong> A {scenario_pct:+d}% reduction would decrease the metric by approximately
            <strong>{abs(avg_impact):.1f} units/week</strong>, representing a total loss of
            <strong>{abs(total_impact):.1f}</strong> units over the {weeks_ahead}-week window.
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# Footer
# =====================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#636e72; font-size:0.8rem;'>"
    "Built with for NatWest 'Code for Purpose' Hackathon &nbsp;|&nbsp; "
    "Powered by Holt-Winters Exponential Smoothing &nbsp;|&nbsp; "
    "Automated Insights Generation Engine"
    "</p>",
    unsafe_allow_html=True,
)
