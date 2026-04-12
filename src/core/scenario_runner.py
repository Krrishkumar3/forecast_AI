"""
scenario_runner.py — What-If Scenario Modelling

Applies a user-defined percentage adjustment to the baseline forecast and
presents both projections side-by-side so stakeholders can quantify the impact
of hypothetical changes (e.g. "+10 % marketing traffic", "−15 % conversion").

This is intentionally a pure arithmetic transformation — no additional model
re-training — so results remain immediately interpretable and deterministic.
"""

import pandas as pd


class ScenarioForecaster:
    """
    Applies percentage-based what-if modifiers to a forecast table and
    produces a comparison view.
    """

    @staticmethod
    def apply_scenario(forecast_df: pd.DataFrame, percentage_change: float) -> pd.DataFrame:
        """
        Scale the 'Likely_Estimate' column by a percentage and return a
        side-by-side comparison.

        Args:
            forecast_df:       Output from Forecaster.generate_forecast().
            percentage_change: Percentage modifier, e.g. 10.0 for +10 %.

        Returns:
            DataFrame with columns:
                Date, Likely_Estimate, Scenario_(<±X>%), Numerical_Impact

        Raises:
            ValueError: If forecast_df lacks the required 'Likely_Estimate' column.
        """
        if 'Likely_Estimate' not in forecast_df.columns:
            raise ValueError("Forecast dataframe must contain a 'Likely_Estimate' column.")

        scenario = forecast_df.copy()

        multiplier = 1 + (percentage_change / 100.0)
        scenario_column_name = f'Scenario_({percentage_change:+.1f}%)'

        scenario[scenario_column_name] = scenario['Likely_Estimate'] * multiplier
        scenario['Numerical_Impact'] = (
            scenario[scenario_column_name] - scenario['Likely_Estimate']
        )

        return scenario[['Date', 'Likely_Estimate', scenario_column_name, 'Numerical_Impact']].round(2)
