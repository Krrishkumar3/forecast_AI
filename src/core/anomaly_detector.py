"""
anomaly_detector.py — Historical Anomaly Detection

Uses a rolling Z-score approach to flag data points that deviate significantly
from the recent local trend.  The rolling window is lagged by one period so
that the current observation does not contaminate its own baseline — a subtle
but critical guard against look-ahead bias.

Threshold guidance:
  - Z > 2.0 → ~5 % of normally-distributed data flagged  (default)
  - Z > 3.0 → ~0.3 % flagged  (stricter, fewer false positives)
"""

import pandas as pd
import numpy as np


class AnomalyDetector:
    """
    Flags historical data points whose values fall significantly outside the
    recent rolling mean, using a configurable Z-score threshold.
    """

    def __init__(self, df: pd.DataFrame, target_col: str, date_col: str = 'date'):
        """
        Args:
            df:         DataFrame containing historical observations.
            target_col: Column with the metric to analyse.
            date_col:   Column with date/time values.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col

        if self.date_col in self.df.columns:
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
            self.df.set_index(self.date_col, inplace=True)

    def detect_anomalies(
        self,
        window_size: int = 4,
        dynamic_z_score_threshold: float = 2.0,
    ) -> pd.DataFrame:
        """
        Scan the historical series and return rows classified as anomalous.

        Args:
            window_size:              Number of past observations for the rolling baseline.
            dynamic_z_score_threshold: Absolute Z-score above which a point is flagged.

        Returns:
            DataFrame containing only the flagged rows, with added columns:
                Rolling_Mean, Rolling_Std, Z_Score, Is_Anomaly
        """
        data = self.df[[self.target_col]].copy()

        # Lag by 1 so the current point does not influence its own baseline
        data['Lagged'] = data[self.target_col].shift(1)
        data['Rolling_Mean'] = data['Lagged'].rolling(window=window_size).mean()
        data['Rolling_Std'] = data['Lagged'].rolling(window=window_size).std()

        # Guard against division by zero in perfectly flat series
        safe_std = data['Rolling_Std'].replace(0, 1e-10)
        data['Z_Score'] = (data[self.target_col] - data['Rolling_Mean']) / safe_std

        data['Is_Anomaly'] = np.abs(data['Z_Score']) > dynamic_z_score_threshold

        anomalies = data[data['Is_Anomaly']].copy().dropna()
        return anomalies
