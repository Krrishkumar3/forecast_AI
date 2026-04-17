"""
backtester.py — Walk-Forward Model Validation Engine

Implements a rigorous backtesting framework that simulates how the forecasting
model would have performed on historical data. This answers the critical
question: "Should I actually trust this model?"

Methodology:
  1. Split the dataset into expanding training windows + holdout periods.
  2. At each step, train the model on all data up to that point and
     forecast the next N periods.
  3. Compare predictions vs. actuals using standard error metrics.

This is the gold standard for time-series model evaluation, far more
credible than a simple train/test split.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class Backtester:
    """
    Walk-forward validation engine that evaluates forecast accuracy
    on historical data using expanding windows.
    """

    def __init__(self, df: pd.DataFrame, target_col: str, date_col: str = 'date'):
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col

        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df = self.df.sort_values(self.date_col)

    def run_backtest(self, holdout_weeks: int = 4, min_train_size: int = 12) -> dict:
        """
        Execute walk-forward backtesting.

        Args:
            holdout_weeks:  Number of periods to forecast at each step.
            min_train_size: Minimum observations before first backtest step.

        Returns:
            dict with:
                'predictions': DataFrame of Date, Actual, Predicted
                'metrics': dict of MAPE, MAE, RMSE, directional_accuracy
                'residuals': Series of prediction errors
        """
        series = self.df[[self.date_col, self.target_col]].copy()
        series = series.set_index(self.date_col)
        values = series[self.target_col].values
        dates = series.index

        all_preds = []
        all_actuals = []
        all_dates = []

        # Walk forward through the data
        n = len(values)
        step_size = max(1, holdout_weeks)

        for split_point in range(min_train_size, n - holdout_weeks + 1, step_size):
            train = values[:split_point]
            actual_holdout = values[split_point:split_point + holdout_weeks]
            holdout_dates = dates[split_point:split_point + holdout_weeks]

            try:
                model = ExponentialSmoothing(
                    train,
                    trend='add',
                    seasonal=None,
                    initialization_method='estimated',
                )
                fitted = model.fit(optimized=True)
                preds = fitted.forecast(holdout_weeks)

                for i in range(len(actual_holdout)):
                    all_dates.append(holdout_dates[i])
                    all_actuals.append(actual_holdout[i])
                    all_preds.append(preds[i])

            except Exception:
                # Skip this window if model fitting fails (e.g. too few points)
                continue

        if not all_preds:
            return {
                'predictions': pd.DataFrame(),
                'metrics': {'mape': None, 'mae': None, 'rmse': None, 'directional_accuracy': None},
                'residuals': pd.Series(dtype=float),
            }

        actuals = np.array(all_actuals)
        preds = np.array(all_preds)
        residuals = actuals - preds

        # --- Metrics ---
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))

        # MAPE: guard against division by zero
        nonzero_mask = actuals != 0
        if nonzero_mask.any():
            mape = np.mean(np.abs(residuals[nonzero_mask] / actuals[nonzero_mask])) * 100
        else:
            mape = None

        # Directional accuracy: did the model predict the right direction of change?
        if len(actuals) > 1:
            actual_dir = np.diff(actuals) > 0
            pred_dir = np.diff(preds) > 0
            directional_accuracy = np.mean(actual_dir == pred_dir) * 100
        else:
            directional_accuracy = None

        predictions_df = pd.DataFrame({
            'Date': all_dates,
            'Actual': actuals,
            'Predicted': np.round(preds, 2),
            'Error': np.round(residuals, 2),
            'Abs_Pct_Error': np.round(np.abs(residuals / np.where(actuals == 0, 1, actuals)) * 100, 2),
        })

        return {
            'predictions': predictions_df,
            'metrics': {
                'mape': round(mape, 2) if mape is not None else None,
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'directional_accuracy': round(directional_accuracy, 1) if directional_accuracy is not None else None,
            },
            'residuals': pd.Series(residuals),
        }

    @staticmethod
    def interpret_mape(mape: float) -> str:
        """Return a human-readable interpretation of MAPE."""
        if mape is None:
            return "Unable to calculate — check for zero values in the data."
        if mape < 5:
            return "Excellent — highly accurate predictions"
        if mape < 10:
            return "Good — reliable for business decisions"
        if mape < 20:
            return "Reasonable — useful with caveats"
        return "Poor — consider additional data or model tuning"

    @staticmethod
    def get_accuracy_grade(mape: float) -> tuple:
        """Return a letter grade and color for the model accuracy."""
        if mape is None:
            return "N/A", "#b2bec3"
        if mape < 5:
            return "A+", "#00b894"
        if mape < 8:
            return "A", "#00cec9"
        if mape < 12:
            return "B", "#fdcb6e"
        if mape < 20:
            return "C", "#e17055"
        return "D", "#d63031"
