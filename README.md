# NatWest Code for Purpose — Advanced Predictive Forecasting

> A lightweight, transparent forecasting tool that transforms historical data into actionable insights — built for the NatWest Group "Code for Purpose" hackathon.

---

## Overview

| | |
|---|---|
| **What** | A full-stack Python forecasting toolkit with an interactive dashboard, REST API, and database layer. It produces short-term predictions, detects anomalies, and models what-if scenarios — all with built-in explainability. |
| **Why** | Decision-makers need forecasts they can *trust and understand*. This tool avoids black-box complexity by pairing every statistical prediction with a simple baseline and plain-English explanations. |
| **Who** | Analysts, product managers, and business stakeholders who need rapid, reliable, and easily explainable insights. |

---

## Features

### 1. Short-Term Forecasting (1–6 weeks)
- Generates a **central "likely" estimate** using Holt-Winters Exponential Smoothing
- Provides **Low / High uncertainty bounds** (90% confidence interval)
- Includes a **simple moving-average baseline** for transparent comparison

### 2. Anomaly Detection
- Scans historical data using a **rolling Z-score** approach to flag unexpected spikes or dips
- Uses **lagged rolling windows** to prevent look-ahead bias
- Generates **automated explanations** via statistical heuristics / LLM API (if configured) — falls back gracefully to rule-based templates

### 3. Scenario Forecasting ("What-If")
- Apply percentage-based adjustments (e.g. *"+10% traffic"*, *"−15% conversion"*)
- View **side-by-side comparison** of baseline vs. scenario with numerical impact

### 4. Interactive Dashboard (Streamlit)
- **Upload CSV** or use bundled sample data
- **Line charts** with shaded confidence bands (Altair)
- **Anomaly cards** with automated insights
- **Real-time What-If slider** that updates projections instantly
- Premium dark glassmorphism UI

### 5. REST API (FastAPI)
- `POST /forecast` — Returns predictions with confidence intervals
- `POST /detect-anomalies` — Returns flagged indices with Z-score rationales
- `POST /scenario` — Returns adjusted projections for what-if analysis
- Auto-generated Swagger docs at `/docs`

### 6. Database Layer (SQLite + SQLAlchemy)
- **metrics** table for time-series data storage
- **forecast_history** table for hold-out validation
- Utility to bridge DB data directly into pandas DataFrames
- Swappable to PostgreSQL via `DATABASE_URL` env var

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Forecasting | `pandas`, `numpy`, `statsmodels` (Holt-Winters) |
| Dashboard | `streamlit`, `altair` |
| REST API | `fastapi`, `uvicorn`, `pydantic` |
| Database | `sqlalchemy`, SQLite (default) / PostgreSQL |
| Explanations | Automated statistical heuristics |
| Environment | `python-dotenv` |
| Testing | `pytest` |

---

## Project Structure

```
forecast_AI/
├── assets/
│   └── sample_data.csv              # 52 weeks of realistic historical data
├── data/
│   └── forecast.db                  # SQLite database (auto-created)
├── src/
│   ├── __init__.py
│   ├── core/                        # Domain logic (framework-agnostic)
│   │   ├── __init__.py
│   │   ├── forecaster.py            # Holt-Winters forecasting engine
│   │   ├── anomaly_detector.py      # Z-score anomaly detection
│   │   ├── scenario_runner.py       # What-if scenario modelling
│   │   └── explainer.py             # Gemini / rule-based explanations
│   ├── api/                         # REST API layer
│   │   ├── __init__.py
│   │   └── app.py                   # FastAPI endpoints
│   ├── db/                          # Database layer
│   │   ├── __init__.py
│   │   └── db_manager.py            # SQLAlchemy models & utilities
│   ├── dashboard.py                 # Streamlit interactive UI
│   └── main.py                      # CLI pipeline orchestrator
├── tests/
│   ├── __init__.py
│   └── test_forecaster.py           # 16 unit tests
├── .env.example                     # Template for API keys
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip

### Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/Krrishkumar3/forecast_AI.git
cd forecast_AI

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Configure environment variables (optional)
cp .env.example .env
# Edit .env to add your GEMINI_API_KEY (optional — tool works without it)
```

---

## Usage

### Option 1: Interactive Dashboard (Recommended)

```bash
streamlit run src/dashboard.py
```

Opens a browser with the full interactive dashboard — upload data, adjust settings, see charts and anomalies in real-time.

### Option 2: REST API

```bash
uvicorn src.api.app:app --reload --port 8000
```

Then visit http://localhost:8000/docs for the interactive Swagger documentation.

**Example API call:**
```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"date": "2023-01-01", "value": 100},
      {"date": "2023-01-08", "value": 105},
      {"date": "2023-01-15", "value": 102},
      {"date": "2023-01-22", "value": 108},
      {"date": "2023-01-29", "value": 107},
      {"date": "2023-02-05", "value": 110},
      {"date": "2023-02-12", "value": 112},
      {"date": "2023-02-19", "value": 109}
    ],
    "weeks_ahead": 4
  }'
```

### Option 3: CLI Pipeline

```bash
python src/main.py
```

### Option 4: Seed the Database

```bash
python src/db/db_manager.py
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## CLI Output Example

```text
============================================================
  NatWest 'Code for Purpose': Advanced Predictive Forecasting
============================================================

Loaded 52 historical records from sample_data.csv

--- 1. SHORT-TERM FORECAST (Next 4 weeks) ---
      Date  Baseline_Avg  Likely_Estimate  Low_Bound  High_Bound
2023-12-31         116.0           114.37      97.39      131.35
2024-01-07         116.0           114.69      97.71      131.67
2024-01-14         116.0           115.01      98.03      131.98
2024-01-21         116.0           115.32      98.34      132.30

--- 2. ANOMALY DETECTION ---
Found 6 anomaly(ies) in the historical dataset!

  > Date: 2023-10-15  |  Value: 160.0  |  Expected: 108.8
    [Rule-Based Explanation] This rapid 47% increase deviates from normal
    trends, likely driven by a temporary promotional campaign or a
    localised surge in demand.

  > Date: 2023-12-24  |  Value: 98.0  |  Expected: 121.5
    [Rule-Based Explanation] This sharp 19% drop represents an unexpected
    anomaly, suggesting a potential temporary outage, data delay, or
    regional holiday impacting normal volumes.

--- 3. SCENARIO FORECASTING (What-if: +15% Volume) ---
      Date  Likely_Estimate  Scenario_(+15.0%)  Numerical_Impact
2023-12-31           114.37             131.53             17.16
2024-01-07           114.69             131.89             17.20
2024-01-14           115.01             132.26             17.25
2024-01-21           115.32             132.62             17.30

Pipeline complete.
```

> **Reading the output:**
> - **Baseline_Avg** = simple 4-week trailing mean (sanity check)
> - **Likely_Estimate** = Holt-Winters model prediction (central estimate)
> - **Low / High Bound** = 90% confidence interval
> - **Scenario** = adjusted projection for the what-if assumption

---

## Security

- All API keys are loaded from **environment variables** via `python-dotenv`
- A `.env.example` is provided as a template — real `.env` files are git-ignored
- **No secrets are hard-coded** anywhere in the codebase
- Database URL is configurable via `DATABASE_URL` env var

---

## Testing

**16 tests** across 4 test classes:

| Class | Tests | What it validates |
|---|---|---|
| `TestForecaster` | 5 | Output shape, columns, bounds, validation |
| `TestAnomalyDetector` | 3 | Spike detection, flat data, Z-score presence |
| `TestScenarioForecaster` | 4 | Arithmetic accuracy, negative scenarios, errors |
| `TestDatabaseManager` | 4 | Table CRUD, CSV seeding, forecast history |

---

## License

This project was built for the **NatWest "Code for Purpose" Hackathon** and is provided under the [MIT License](https://opensource.org/licenses/MIT).

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-improvement`)
3. Commit your changes (`git commit -m 'Add some improvement'`)
4. Push to the branch (`git push origin feature/my-improvement`)
5. Open a Pull Request

---

*Built with for NatWest Code for Purpose*
