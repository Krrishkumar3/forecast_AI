"""
trend_analyzer.py — Trend Decomposition & Autocorrelation Analysis

Breaks a time series into its constituent components (trend, seasonal,
residual) using classical additive decomposition, and computes the
autocorrelation function (ACF) for lag analysis.

This gives analysts a deeper understanding of *why* a metric behaves
the way it does, rather than just *what* it will do next.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf


class TrendAnalyzer:
    """
    Performs additive time-series decomposition and autocorrelation analysis.
    """

    def __init__(self, df: pd.DataFrame, target_col: str, date_col: str = 'date'):
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col

        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df = self.df.sort_values(self.date_col).set_index(self.date_col)

    def decompose(self, period: int = 4) -> dict:
        """
        Decompose the time series into trend, seasonal, and residual components.

        Args:
            period: The seasonal period (default 4 for weekly data with
                    monthly seasonality).

        Returns:
            dict with keys: 'trend', 'seasonal', 'residual', 'observed',
            each containing a pandas Series.
        """
        series = self.df[self.target_col].dropna()

        # Ensure we have enough data points for decomposition
        if len(series) < 2 * period:
            period = max(2, len(series) // 4)

        result = seasonal_decompose(series, model='additive', period=period)

        return {
            'observed': result.observed,
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.resid,
        }

    def compute_acf(self, nlags: int = 20) -> pd.DataFrame:
        """
        Compute the autocorrelation function for the time series.

        Args:
            nlags: Number of lag periods to compute.

        Returns:
            DataFrame with columns: Lag, ACF, Significance_Bound
        """
        series = self.df[self.target_col].dropna()
        nlags = min(nlags, len(series) // 2 - 1)

        acf_values = acf(series, nlags=nlags, fft=True)

        # Bartlett's formula for 95% confidence interval
        sig_bound = 1.96 / np.sqrt(len(series))

        return pd.DataFrame({
            'Lag': range(nlags + 1),
            'ACF': acf_values,
            'Significance_Upper': sig_bound,
            'Significance_Lower': -sig_bound,
        })

    def compute_summary_stats(self) -> dict:
        """
        Return a rich set of descriptive statistics about the series.
        """
        series = self.df[self.target_col].dropna()

        # Compute week-over-week changes
        pct_changes = series.pct_change().dropna()

        # Volatility (standard deviation of % changes)
        volatility = pct_changes.std() * 100

        # Trend direction over last N points
        recent = series.tail(8)
        slope = np.polyfit(range(len(recent)), recent.values, 1)[0]

        return {
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'volatility_pct': volatility,
            'trend_slope': slope,
            'trend_direction': 'Upward' if slope > 0.5 else ('Downward' if slope < -0.5 else 'Flat'),
            'total_observations': len(series),
        }
