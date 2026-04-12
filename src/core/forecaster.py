"""
forecaster.py — Short-Term Forecasting Engine

Produces 1–6 week projections with:
  • A central "likely" estimate via Holt-Winters Exponential Smoothing
  • A simple moving-average baseline for transparency & overfitting guard
  • Low / High uncertainty bounds derived from historical residual variance

Design rationale:
  - Seasonal component is intentionally disabled (`seasonal=None`) because the
    sample dataset is small.  With larger data (>2 full cycles) the seasonal
    parameter can safely be switched to 'add'.
  - The 90 % confidence interval (z ≈ 1.645) balances informativeness without
    being so wide it loses practical value.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class Forecaster:
    """
    Generates transparent short-term forecasts with uncertainty bands.

    Every forecast is paired with a simple moving-average baseline so
    stakeholders can judge whether the model adds value over a naïve approach.
    """

    def __init__(self, df: pd.DataFrame, target_col: str, date_col: str = 'date'):
        """
        Args:
            df:         DataFrame with at least a date column and a numeric target.
            target_col: Column name of the metric to forecast (e.g. 'traffic').
            date_col:   Column name containing timestamps.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col

        # Parse dates and enforce chronological order
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df = self.df.sort_values(self.date_col).set_index(self.date_col)

    def generate_forecast(self, weeks_ahead: int = 4, seasonal_periods: int = 4) -> pd.DataFrame:
        """
        Produce a tabular forecast comparing a simple baseline against the model.

        Args:
            weeks_ahead:      Number of future weekly periods (1–6).
            seasonal_periods: Cycle length passed to Holt-Winters. Not used when
                              seasonal component is disabled; kept as a parameter
                              for easy future upgrades.

        Returns:
            DataFrame with columns:
                Date, Baseline_Avg, Likely_Estimate, Low_Bound, High_Bound

        Raises:
            ValueError: If weeks_ahead is outside the 1–6 range.
        """
        if weeks_ahead < 1 or weeks_ahead > 6:
            raise ValueError("Forecast horizon must be between 1 and 6 weeks for model stability.")

        time_series = self.df[self.target_col]

        # ----- Simple Baseline: 4-week trailing mean -----
        # Provides a transparent sanity-check against the statistical model.
        trailing_mean = time_series.tail(4).mean()
        baseline_values = [trailing_mean] * weeks_ahead

        # ----- Statistical Model: Additive-trend Exponential Smoothing -----
        model = ExponentialSmoothing(
            time_series,
            trend='add',
            seasonal=None,  # disabled to prevent overfitting on small samples
            initialization_method='estimated',
        )
        fitted_model = model.fit()
        forecast_values = fitted_model.forecast(weeks_ahead)

        # ----- Uncertainty bounds from historical residuals -----
        residual_std = np.std(fitted_model.resid)
        margin_of_error = 1.645 * residual_std  # ≈ 90 % confidence interval

        # ----- Build future date index -----
        last_date = time_series.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=7),
            periods=weeks_ahead,
            freq='W',
        )

        results = pd.DataFrame({
            'Date': future_dates,
            'Baseline_Avg': baseline_values,
            'Likely_Estimate': forecast_values.values,
            'Low_Bound': (forecast_values.values - margin_of_error).clip(min=0),
            'High_Bound': forecast_values.values + margin_of_error,
        })

        # Round for readability by non-technical stakeholders
        numeric_columns = ['Baseline_Avg', 'Likely_Estimate', 'Low_Bound', 'High_Bound']
        results[numeric_columns] = results[numeric_columns].round(2)

        return results
