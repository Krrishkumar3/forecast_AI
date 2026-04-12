"""
db_manager.py — SQLite database layer using SQLAlchemy.

Provides a lightweight persistence layer for:
  • metrics       — raw time-series observations (date, metric_name, value)
  • forecast_history — past predictions for hold-out validation & auditing

Design rationale:
  - SQLite is used because it is zero-config, file-based, and perfect for a
    hackathon demo.  Swapping to PostgreSQL requires only changing the
    DATABASE_URL environment variable.
  - SQLAlchemy ORM is used so the schema is version-controlled in code.
  - A utility function (`load_metrics_as_dataframe`) provides a direct bridge
    from the DB into the pandas DataFrame expected by the Forecaster class.

Usage:
    from src.db.db_manager import DatabaseManager

    db = DatabaseManager()              # uses default SQLite path
    db.create_tables()                  # one-time setup
    db.seed_from_csv("assets/sample_data.csv")
    df = db.load_metrics_as_dataframe(metric_name="traffic")
"""

import os
import sys
from datetime import datetime, date
from typing import Optional, List

import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    Date,
    DateTime,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# SQLAlchemy setup
# ---------------------------------------------------------------------------
Base = declarative_base()

# Default database path — stored in the /data directory at project root
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "forecast.db")
DEFAULT_DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")


# =====================================================================
# ORM Models
# =====================================================================

class Metric(Base):
    """
    Stores raw time-series observations.

    Each row is one data point for a named metric on a specific date.
    The (date, metric_name) pair is unique to prevent duplicate imports.
    """
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    metric_name = Column(String(100), nullable=False, index=True, default="traffic")
    value = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("date", "metric_name", name="uq_date_metric"),
    )

    def __repr__(self):
        return f"<Metric(date={self.date}, metric={self.metric_name}, value={self.value})>"


class ForecastHistory(Base):
    """
    Stores past predictions for hold-out validation and auditing.

    By comparing stored forecasts against actual future values, we can
    measure real-world accuracy over time and detect model drift.
    """
    __tablename__ = "forecast_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    forecast_date = Column(Date, nullable=False, doc="The date this prediction targets")
    metric_name = Column(String(100), nullable=False, default="traffic")
    baseline_avg = Column(Float, nullable=False, doc="Simple moving average baseline")
    likely_estimate = Column(Float, nullable=False, doc="Holt-Winters central estimate")
    low_bound = Column(Float, nullable=False, doc="Lower 90% confidence bound")
    high_bound = Column(Float, nullable=False, doc="Upper 90% confidence bound")
    model_version = Column(String(50), default="holt_winters_v1", doc="Model identifier for tracking")
    created_at = Column(DateTime, default=datetime.utcnow, doc="When this forecast was generated")

    def __repr__(self):
        return (
            f"<ForecastHistory(target={self.forecast_date}, "
            f"estimate={self.likely_estimate}, range=[{self.low_bound}, {self.high_bound}])>"
        )


# =====================================================================
# Database Manager
# =====================================================================

class DatabaseManager:
    """
    High-level interface for database operations.

    Handles table creation, data seeding, metric insertion, and the
    critical bridge function that feeds data into the Forecaster.
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Args:
            database_url: SQLAlchemy connection string.
                          Defaults to sqlite:///data/forecast.db
        """
        self.database_url = database_url or DEFAULT_DATABASE_URL
        self.engine = create_engine(self.database_url, echo=False)
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self):
        """Create all tables if they don't exist. Safe to call repeatedly."""
        Base.metadata.create_all(self.engine)

    def drop_tables(self):
        """Drop all tables. Use with caution — intended for test teardown."""
        Base.metadata.drop_all(self.engine)

    # -----------------------------------------------------------------
    # Metrics CRUD
    # -----------------------------------------------------------------

    def insert_metric(self, metric_date: date, value: float, metric_name: str = "traffic"):
        """Insert a single metric observation, ignoring duplicates."""
        session = self.Session()
        try:
            existing = (
                session.query(Metric)
                .filter_by(date=metric_date, metric_name=metric_name)
                .first()
            )
            if existing:
                existing.value = value  # update if date already exists
            else:
                session.add(Metric(date=metric_date, metric_name=metric_name, value=value))
            session.commit()
        finally:
            session.close()

    def seed_from_csv(self, csv_path: str, metric_name: str = "traffic"):
        """
        Bulk-import a CSV file into the metrics table.

        Expected CSV columns: date, <metric_name>
        Existing records are updated (upsert behaviour).
        """
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        session = self.Session()
        try:
            for _, row in df.iterrows():
                existing = (
                    session.query(Metric)
                    .filter_by(date=row["date"], metric_name=metric_name)
                    .first()
                )
                if existing:
                    existing.value = float(row[metric_name])
                else:
                    session.add(Metric(
                        date=row["date"],
                        metric_name=metric_name,
                        value=float(row[metric_name]),
                    ))
            session.commit()
            print(f"Seeded {len(df)} records into '{metric_name}' metrics table.")
        finally:
            session.close()

    def load_metrics_as_dataframe(self, metric_name: str = "traffic") -> pd.DataFrame:
        """
        Load all observations for a metric into a pandas DataFrame.

        Returns a DataFrame with columns ['date', <metric_name>] sorted
        chronologically — ready to pass directly into the Forecaster class.
        """
        session = self.Session()
        try:
            rows = (
                session.query(Metric.date, Metric.value)
                .filter(Metric.metric_name == metric_name)
                .order_by(Metric.date)
                .all()
            )
            df = pd.DataFrame(rows, columns=["date", metric_name])
            df["date"] = pd.to_datetime(df["date"])
            return df
        finally:
            session.close()

    # -----------------------------------------------------------------
    # Forecast History
    # -----------------------------------------------------------------

    def save_forecast(self, forecast_df: pd.DataFrame, metric_name: str = "traffic"):
        """
        Persist a forecast DataFrame into the forecast_history table.

        This enables retrospective accuracy analysis: compare stored
        predictions against actual values that arrive later.
        """
        session = self.Session()
        try:
            for _, row in forecast_df.iterrows():
                record = ForecastHistory(
                    forecast_date=row["Date"].date() if hasattr(row["Date"], "date") else row["Date"],
                    metric_name=metric_name,
                    baseline_avg=float(row["Baseline_Avg"]),
                    likely_estimate=float(row["Likely_Estimate"]),
                    low_bound=float(row["Low_Bound"]),
                    high_bound=float(row["High_Bound"]),
                )
                session.add(record)
            session.commit()
            print(f"Saved {len(forecast_df)} forecast records to history.")
        finally:
            session.close()

    def load_forecast_history(self, metric_name: str = "traffic") -> pd.DataFrame:
        """Retrieve all stored forecasts for a metric as a DataFrame."""
        session = self.Session()
        try:
            rows = (
                session.query(
                    ForecastHistory.forecast_date,
                    ForecastHistory.baseline_avg,
                    ForecastHistory.likely_estimate,
                    ForecastHistory.low_bound,
                    ForecastHistory.high_bound,
                    ForecastHistory.model_version,
                    ForecastHistory.created_at,
                )
                .filter(ForecastHistory.metric_name == metric_name)
                .order_by(ForecastHistory.forecast_date)
                .all()
            )
            return pd.DataFrame(rows, columns=[
                "forecast_date", "baseline_avg", "likely_estimate",
                "low_bound", "high_bound", "model_version", "created_at",
            ])
        finally:
            session.close()


# =====================================================================
# CLI entry point — seed the database from the sample CSV
# =====================================================================

if __name__ == "__main__":
    csv_path = os.path.join(PROJECT_ROOT, "assets", "sample_data.csv")

    if not os.path.exists(csv_path):
        print(f"[ERROR] Sample data not found at: {csv_path}")
        sys.exit(1)

    db = DatabaseManager()
    db.create_tables()
    db.seed_from_csv(csv_path, metric_name="traffic")

    # Verify the load
    df = db.load_metrics_as_dataframe("traffic")
    print(f"Loaded {len(df)} records from database:")
    print(df.head(10).to_string(index=False))
