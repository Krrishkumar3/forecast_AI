"""
main.py — NatWest 'Code for Purpose' AI Predictive Forecasting Pipeline

Orchestrates the three core use-cases:
  1. Short-term forecasting with uncertainty bands
  2. Historical anomaly detection with AI-powered explanations
  3. What-if scenario modelling compared side-by-side with baseline

Run from the project root:
    python src/main.py
"""

import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the /src directory is on the import path regardless of how the user
# invokes the script (e.g. `python src/main.py` from project root).
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from forecaster import Forecaster
from anomaly_detector import AnomalyDetector
from scenario_runner import ScenarioForecaster
from explainer import AnomalyExplainer


def _resolve_data_path() -> str:
    """Return the absolute path to the sample dataset in /assets."""
    return os.path.join(SRC_DIR, '..', 'assets', 'sample_data.csv')


def run_pipeline():
    """
    Execute the full forecasting pipeline end-to-end.

    Steps:
        1. Load historical data from CSV.
        2. Produce a short-term forecast with low/likely/high bands.
        3. Detect anomalies in the historical record.
        4. Generate a what-if scenario and compare against baseline.
    """
    print("=" * 60)
    print("  NatWest 'Code for Purpose': AI Predictive Forecasting")
    print("=" * 60, "\n")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data_path = _resolve_data_path()
    if not os.path.exists(data_path):
        print(f"[ERROR] Could not find dataset at:\n  {data_path}")
        print("Please ensure the file exists in the /assets directory.")
        sys.exit(1)

    historical_data = pd.read_csv(data_path)
    target_metric = 'traffic'
    print(f"Loaded {len(historical_data)} historical records from {os.path.basename(data_path)}\n")

    # ------------------------------------------------------------------
    # 1. Short-Term Forecasting
    # ------------------------------------------------------------------
    print("--- 1. SHORT-TERM FORECAST (Next 4 weeks) ---")
    forecaster = Forecaster(historical_data, target_col=target_metric, date_col='date')
    forecast_df = forecaster.generate_forecast(weeks_ahead=4)
    print(forecast_df.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # 2. Anomaly Detection & AI Explanation
    # ------------------------------------------------------------------
    print("--- 2. ANOMALY DETECTION ---")
    detector = AnomalyDetector(historical_data, target_col=target_metric, date_col='date')
    anomalies = detector.detect_anomalies(window_size=4, dynamic_z_score_threshold=2.5)

    if anomalies.empty:
        print("No significant anomalies detected in historical data.\n")
    else:
        print(f"Found {len(anomalies)} anomaly(ies) in the historical dataset!\n")
        explainer = AnomalyExplainer()

        for date_idx, row in anomalies.iterrows():
            date_str = str(date_idx.date()) if hasattr(date_idx, 'date') else str(date_idx)
            actual_value = row[target_metric]
            expected_value = row['Rolling_Mean']
            z_score = row['Z_Score']

            print(f"  > Date: {date_str}  |  Value: {actual_value:.1f}  |  Expected: {expected_value:.1f}")

            explanation = explainer.generate_explanation(
                date=date_str,
                actual_value=actual_value,
                expected_value=expected_value,
                z_score=z_score,
            )
            print(f"    {explanation}\n")

    # ------------------------------------------------------------------
    # 3. Scenario Forecasting
    # ------------------------------------------------------------------
    scenario_pct = 15.0
    print(f"--- 3. SCENARIO FORECASTING (What-if: +{scenario_pct:.0f}% Volume) ---")
    scenario_df = ScenarioForecaster.apply_scenario(forecast_df, percentage_change=scenario_pct)
    print(scenario_df.to_string(index=False))
    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
