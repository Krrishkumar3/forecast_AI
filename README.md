# 🏦 NatWest Code for Purpose — AI Predictive Forecasting

> A lightweight, transparent AI forecasting tool that transforms historical data into actionable insights — built for the NatWest Group "Code for Purpose" hackathon.

---

## 📖 Overview

| | |
|---|---|
| **What** | A Python-based forecasting toolkit that produces short-term predictions, detects anomalies, and models what-if scenarios — all with built-in explainability. |
| **Why** | Decision-makers need forecasts they can *trust and understand*. This tool avoids black-box complexity by pairing every AI prediction with a simple baseline and plain-English explanations. |
| **Who** | Analysts, product managers, and business stakeholders who need rapid, reliable, and easily explainable insights. |

---

## ✨ Features

### 1. Short-Term Forecasting (1–6 weeks)
- Generates a **central "likely" estimate** using Holt-Winters Exponential Smoothing
- Provides **Low / High uncertainty bounds** (90% confidence interval) so stakeholders see the range of possibilities
- Includes a **simple moving-average baseline** for transparent comparison — if the model can't beat a basic average, you'll know immediately

### 2. Anomaly Detection
- Scans historical data using a **rolling Z-score** approach to flag unexpected spikes or dips
- Uses **lagged rolling windows** to prevent look-ahead bias
- Generates **AI-powered explanations** via Google Gemini (free tier) — or falls back gracefully to rule-based templates if no API key is configured

### 3. Scenario Forecasting ("What-If")
- Apply percentage-based adjustments (e.g. *"+10% traffic"*, *"−15% conversion"*) to the baseline forecast
- View **side-by-side comparison** of baseline vs. scenario with numerical impact
- Pure arithmetic transformation — results are immediately interpretable

### 4. Anti-Overfitting by Design
- Every forecast is compared against a **4-week trailing moving average**
- Seasonal component is disabled for small datasets to prevent spurious patterns
- Residual-based uncertainty bands honestly reflect model limitations

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Data & Forecasting | `pandas`, `numpy`, `statsmodels` (Holt-Winters) |
| AI Explanations | `google-generativeai` (Gemini Free Tier) |
| Environment Mgmt | `python-dotenv` |
| Testing | `unittest` (stdlib) |

---

## 📁 Project Structure

```
forecast_AI/
├── assets/
│   └── sample_data.csv          # 52 weeks of realistic historical data
├── src/
│   ├── __init__.py
│   ├── main.py                  # Pipeline orchestrator
│   ├── forecaster.py            # Holt-Winters forecasting engine
│   ├── anomaly_detector.py      # Z-score anomaly detection
│   ├── scenario_runner.py       # What-if scenario modelling
│   └── explainer.py             # Gemini / rule-based explanation generator
├── tests/
│   ├── __init__.py
│   └── test_forecaster.py       # Unit tests for all modules
├── .env.example                 # Template for API keys (never commit .env)
├── .gitignore                   # Excludes venv, .env, __pycache__
├── requirements.txt             # One-command dependency install
└── README.md                    # This file
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/Krrishkumar3/forecast_AI.git
cd forecast_AI

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env and paste your Gemini API key (optional — tool works without it)
```

> **Note:** The tool runs fully offline using rule-based explanations if no `GEMINI_API_KEY` is set. The Gemini integration is an enhancement, not a requirement.

---

## ▶️ Usage

### Run the Full Pipeline

```bash
python src/main.py
```

### Run the Tests

```bash
python -m pytest tests/ -v
# or
python -m unittest discover -s tests -v
```

---

## 📊 Example Output

```text
============================================================
  NatWest 'Code for Purpose': AI Predictive Forecasting
============================================================

Loaded 52 historical records from sample_data.csv

--- 1. SHORT-TERM FORECAST (Next 4 weeks) ---
       Date  Baseline_Avg  Likely_Estimate  Low_Bound  High_Bound
 2023-12-31        116.00           112.47      99.84      125.10
 2024-01-07        116.00           111.82      99.19      124.45
 2024-01-14        116.00           111.17      98.54      123.80
 2024-01-21        116.00           110.52      97.89      123.15

--- 2. ANOMALY DETECTION ---
Found 2 anomaly(ies) in the historical dataset!

  > Date: 2023-10-15  |  Value: 160.0  |  Expected: 106.2
    [Rule-Based Explanation] This rapid 51% increase deviates from normal
    trends, likely driven by a temporary promotional campaign or a
    localised surge in demand.

  > Date: 2023-12-24  |  Value: 98.0  |  Expected: 122.0
    [Rule-Based Explanation] This sharp 20% drop represents an unexpected
    anomaly, suggesting a potential temporary outage, data delay, or
    regional holiday impacting normal volumes.

--- 3. SCENARIO FORECASTING (What-if: +15% Volume) ---
       Date  Likely_Estimate  Scenario_(+15.0%)  Numerical_Impact
 2023-12-31           112.47             129.34             16.87
 2024-01-07           111.82             128.59             16.77
 2024-01-14           111.17             127.85             16.68
 2024-01-21           110.52             127.10             16.58

✅ Pipeline complete.
```

> **Reading the output:**
> - **Baseline_Avg** = simple 4-week trailing mean (sanity check)
> - **Likely_Estimate** = Holt-Winters model prediction (central estimate)
> - **Low_Bound / High_Bound** = 90% confidence interval
> - **Scenario** = adjusted projection for the what-if assumption

---

## 🔒 Security

- All API keys are loaded from **environment variables** via `python-dotenv`
- A `.env.example` is provided as a template — real `.env` files are git-ignored
- **No secrets are hard-coded** anywhere in the codebase

---

## 🧪 Testing

The test suite validates:

| Test | What it checks |
|---|---|
| `test_forecast_returns_correct_number_of_rows` | Output length matches requested horizon |
| `test_forecast_contains_required_columns` | All transparency columns are present |
| `test_low_bound_never_negative` | No illogical negative traffic predictions |
| `test_high_bound_exceeds_low_bound` | Uncertainty bands are correctly ordered |
| `test_invalid_horizon_raises_error` | Rejects out-of-range forecast requests |
| `test_detects_injected_spike` | Catches an obvious anomaly in synthetic data |
| `test_no_anomalies_in_flat_data` | Doesn't false-alarm on constant series |
| `test_scenario_arithmetic_accuracy` | +10% multiplier is mathematically correct |
| `test_negative_scenario` | Negative scenarios reduce the estimate |
| `test_missing_column_raises_error` | Validates input before processing |

---

## 📝 License

This project was built for the **NatWest "Code for Purpose" Hackathon** and is provided under the [MIT License](https://opensource.org/licenses/MIT).

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-improvement`)
3. Commit your changes (`git commit -m 'Add some improvement'`)
4. Push to the branch (`git push origin feature/my-improvement`)
5. Open a Pull Request

---

*Built with ❤️ for NatWest Code for Purpose*
