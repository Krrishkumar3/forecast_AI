"""
test_forecaster.py — Unit tests for core forecasting modules.

Tests cover:
  • Forecaster output shape, column presence, and bound sanity
  • AnomalyDetector correctly flags injected spikes
  • ScenarioForecaster arithmetic accuracy
  • DatabaseManager CRUD operations
"""

import unittest
import sys
import os
import tempfile

import pandas as pd
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.forecaster import Forecaster
from src.core.anomaly_detector import AnomalyDetector
from src.core.scenario_runner import ScenarioForecaster
from src.db.db_manager import DatabaseManager


class TestForecaster(unittest.TestCase):
    """Validates the short-term forecasting engine."""

    def setUp(self):
        """Create a small but realistic weekly dataset for testing."""
        dates = pd.date_range(start='2023-01-01', periods=12, freq='W-SUN')
        values = [100, 102, 101, 105, 103, 107, 108, 106, 110, 112, 111, 115]
        self.df = pd.DataFrame({'date': dates, 'traffic': values})

    def test_forecast_returns_correct_number_of_rows(self):
        """Each requested week must produce exactly one forecast row."""
        forecaster = Forecaster(self.df, target_col='traffic', date_col='date')
        result = forecaster.generate_forecast(weeks_ahead=3)
        self.assertEqual(len(result), 3)

    def test_forecast_contains_required_columns(self):
        """Output must include baseline, likely estimate, and uncertainty bounds."""
        forecaster = Forecaster(self.df, target_col='traffic', date_col='date')
        result = forecaster.generate_forecast(weeks_ahead=4)

        expected_columns = {'Date', 'Baseline_Avg', 'Likely_Estimate', 'Low_Bound', 'High_Bound'}
        self.assertTrue(expected_columns.issubset(set(result.columns)))

    def test_low_bound_never_negative(self):
        """Negative traffic makes no business sense — low bound should be clipped at 0."""
        forecaster = Forecaster(self.df, target_col='traffic', date_col='date')
        result = forecaster.generate_forecast(weeks_ahead=6)
        self.assertTrue((result['Low_Bound'] >= 0).all())

    def test_high_bound_exceeds_low_bound(self):
        """The high bound must always be greater than or equal to the low bound."""
        forecaster = Forecaster(self.df, target_col='traffic', date_col='date')
        result = forecaster.generate_forecast(weeks_ahead=4)
        self.assertTrue((result['High_Bound'] >= result['Low_Bound']).all())

    def test_invalid_horizon_raises_error(self):
        """Horizons outside 1–6 weeks should be rejected to prevent unreliable projections."""
        forecaster = Forecaster(self.df, target_col='traffic', date_col='date')
        with self.assertRaises(ValueError):
            forecaster.generate_forecast(weeks_ahead=0)
        with self.assertRaises(ValueError):
            forecaster.generate_forecast(weeks_ahead=7)


class TestAnomalyDetector(unittest.TestCase):
    """Validates the anomaly detection Z-score logic."""

    def setUp(self):
        """Create data with a known, injected anomaly spike at week 10."""
        dates = pd.date_range(start='2023-01-01', periods=12, freq='W-SUN')
        # Steady ~100-range data with a massive spike (200) at index 9
        values = [100, 102, 101, 105, 103, 104, 102, 106, 105, 200, 103, 101]
        self.df = pd.DataFrame({'date': dates, 'traffic': values})

    def test_detects_injected_spike(self):
        """The detector must flag the obvious 200-spike in otherwise flat data."""
        detector = AnomalyDetector(self.df, target_col='traffic', date_col='date')
        anomalies = detector.detect_anomalies(window_size=4, dynamic_z_score_threshold=2.0)
        self.assertGreaterEqual(len(anomalies), 1, "Should detect at least one anomaly")

    def test_no_anomalies_in_flat_data(self):
        """Perfectly flat data should produce zero anomalies."""
        dates = pd.date_range(start='2023-01-01', periods=12, freq='W-SUN')
        flat_values = [100] * 12
        flat_df = pd.DataFrame({'date': dates, 'traffic': flat_values})

        detector = AnomalyDetector(flat_df, target_col='traffic', date_col='date')
        anomalies = detector.detect_anomalies(window_size=4, dynamic_z_score_threshold=2.0)
        self.assertEqual(len(anomalies), 0, "Flat data should have no anomalies")

    def test_anomaly_row_contains_z_score(self):
        """Every flagged anomaly must carry its Z-score for downstream explanation."""
        detector = AnomalyDetector(self.df, target_col='traffic', date_col='date')
        anomalies = detector.detect_anomalies()
        if not anomalies.empty:
            self.assertIn('Z_Score', anomalies.columns)


class TestScenarioForecaster(unittest.TestCase):
    """Validates what-if scenario arithmetic."""

    def setUp(self):
        """Generate a forecast table to use as input."""
        dates = pd.date_range(start='2023-01-01', periods=12, freq='W-SUN')
        values = [100, 102, 101, 105, 103, 107, 108, 106, 110, 112, 111, 115]
        df = pd.DataFrame({'date': dates, 'traffic': values})

        forecaster = Forecaster(df, target_col='traffic', date_col='date')
        self.forecast = forecaster.generate_forecast(weeks_ahead=2)

    def test_scenario_column_exists(self):
        """The scenario column must be dynamically named with the percentage."""
        scenario = ScenarioForecaster.apply_scenario(self.forecast, percentage_change=10.0)
        self.assertIn('Scenario_(+10.0%)', scenario.columns)
        self.assertIn('Numerical_Impact', scenario.columns)

    def test_scenario_arithmetic_accuracy(self):
        """The scenario value must equal Likely_Estimate × (1 + pct/100)."""
        scenario = ScenarioForecaster.apply_scenario(self.forecast, percentage_change=10.0)

        expected_value = round(self.forecast.iloc[0]['Likely_Estimate'] * 1.10, 2)
        actual_value = scenario.iloc[0]['Scenario_(+10.0%)']
        self.assertAlmostEqual(actual_value, expected_value, places=1)

    def test_negative_scenario(self):
        """A negative percentage should reduce the estimate correctly."""
        scenario = ScenarioForecaster.apply_scenario(self.forecast, percentage_change=-20.0)
        self.assertIn('Scenario_(-20.0%)', scenario.columns)

        for _, row in scenario.iterrows():
            self.assertLess(row['Numerical_Impact'], 0)

    def test_missing_column_raises_error(self):
        """Passing a DataFrame without 'Likely_Estimate' must raise ValueError."""
        bad_df = pd.DataFrame({'Date': ['2023-01-01'], 'wrong_col': [100]})
        with self.assertRaises(ValueError):
            ScenarioForecaster.apply_scenario(bad_df, percentage_change=10.0)


class TestDatabaseManager(unittest.TestCase):
    """Validates the SQLite database layer."""

    def setUp(self):
        """Create a temporary database for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_forecast.db")
        self.db = DatabaseManager(database_url=f"sqlite:///{self.db_path}")
        self.db.create_tables()

    def tearDown(self):
        """Clean up the temporary database."""
        self.db.drop_tables()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_create_tables_no_error(self):
        """Table creation should complete without errors (idempotent)."""
        self.db.create_tables()  # calling twice should be safe

    def test_insert_and_load_metric(self):
        """A metric inserted via insert_metric must appear in load_metrics_as_dataframe."""
        from datetime import date
        self.db.insert_metric(date(2023, 1, 1), 100.0, "traffic")
        self.db.insert_metric(date(2023, 1, 8), 105.0, "traffic")

        df = self.db.load_metrics_as_dataframe("traffic")
        self.assertEqual(len(df), 2)
        self.assertIn("traffic", df.columns)
        self.assertEqual(df.iloc[0]["traffic"], 100.0)

    def test_seed_from_csv(self):
        """Seeding from the sample CSV must populate the metrics table."""
        csv_path = os.path.join(PROJECT_ROOT, "assets", "sample_data.csv")
        if os.path.exists(csv_path):
            self.db.seed_from_csv(csv_path, metric_name="traffic")
            df = self.db.load_metrics_as_dataframe("traffic")
            self.assertGreater(len(df), 0)

    def test_save_and_load_forecast_history(self):
        """Saving a forecast must persist it and be retrievable."""
        dates = pd.date_range(start='2023-01-01', periods=12, freq='W-SUN')
        values = [100, 102, 101, 105, 103, 107, 108, 106, 110, 112, 111, 115]
        df = pd.DataFrame({'date': dates, 'traffic': values})

        forecaster = Forecaster(df, target_col='traffic', date_col='date')
        forecast_df = forecaster.generate_forecast(weeks_ahead=2)

        self.db.save_forecast(forecast_df, metric_name="traffic")
        history = self.db.load_forecast_history("traffic")
        self.assertEqual(len(history), 2)


if __name__ == '__main__':
    unittest.main()
