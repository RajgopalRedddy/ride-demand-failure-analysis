# When Do Models Fail? Discovering Hidden Failure Patterns in Spatio-Temporal Ride Demand Prediction

## Team
- **Siva Rama Krishna Kasyap Sonthi** — SS24BW
- **Rajgopal Reddy Musku** — RM24M

---

## Overview

This project goes beyond standard ride demand prediction to analyze **when and where** machine learning models fail. Using NYC Yellow Taxi data (2019), we train three models (Linear Regression, Random Forest, LSTM), identify the top 15% worst predictions as "failure cases," and uncover systematic patterns in these failures across time, space, and demand levels.

## Key Questions
1. Which **hours and days** produce the most prediction failures?
2. Which **taxi zones** are consistently hard to predict?
3. Do failures concentrate at **high demand** or **low demand** levels?
4. **Why** do models fail? (SHAP explainability)
5. Are failures **model-specific** or shared across all approaches?

---

## Project Structure

```
ride-demand-failure-analysis/
├── main.py                          # End-to-end pipeline runner
├── requirements.txt                 # Python dependencies
├── README.md
│
├── configs/
│   └── config.yaml                  # All project parameters
│
├── src/                             # Source modules
│   ├── __init__.py
│   ├── data_ingestion.py            # Download & aggregate NYC taxi data
│   ├── feature_engineering.py       # Temporal, lag, rolling features
│   ├── models.py                    # LR, Random Forest, LSTM training
│   ├── failure_analysis.py          # Error analysis & failure detection
│   ├── explainability.py            # SHAP-based feature attribution
│   └── visualizations.py           # All charts, heatmaps, plots
│
├── notebooks/                       # Jupyter notebooks (step-by-step)
│   ├── 01_data_exploration.ipynb    # EDA & data understanding
│   ├── 02_model_training.ipynb      # Train & compare models
│   ├── 03_failure_analysis.ipynb    # Core failure pattern analysis
│   └── 04_shap_explainability.ipynb # SHAP feature importance
│
├── data/                            # Data directory (auto-populated)
│   ├── raw/                         # Downloaded parquet files
│   └── processed/                   # Aggregated & engineered features
│
└── outputs/                         # Results (auto-populated)
    ├── figures/                     # All generated plots
    ├── models/                      # Saved model artifacts
    └── reports/                     # Metrics CSVs & error reports
```

---

## Setup & Installation

```bash
cd ride-demand-failure-analysis

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.9+
- ~10 GB disk space for raw data (12 months of NYC taxi data)
- ~8 GB RAM recommended for full dataset processing

---

## Usage

### Option 1: Full Pipeline (CLI)

```bash
# Full pipeline (downloads data, trains models, generates all outputs)
python main.py

# Skip download if data already exists
python main.py --skip-download

# Use 50 sampled zones for faster experimentation
python main.py --sample-zones 50

# Skip LSTM training (faster, still runs LR + RF)
python main.py --skip-lstm
```

### Option 2: Step-by-Step Notebooks

Run notebooks in order from the `notebooks/` directory:

1. **01_data_exploration.ipynb** — Download data, EDA, visualize demand patterns
2. **02_model_training.ipynb** — Train all three models, compare metrics
3. **03_failure_analysis.ipynb** — Identify and analyze failure patterns
4. **04_shap_explainability.ipynb** — SHAP analysis of why failures occur

### Option 3: Individual Modules

```python
from src.data_ingestion import load_config, download_taxi_data, aggregate_demand
from src.feature_engineering import engineer_features
from src.models import run_all_models

config = load_config('configs/config.yaml')
download_taxi_data(config)
demand = aggregate_demand(config)
features = engineer_features(config)
results = run_all_models(config)
```

---

## Dataset

**NYC Taxi & Limousine Commission (TLC) Trip Record Data**
- Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Period: January – December 2019
- Type: Yellow Taxi trip records
- Key fields: `tpep_pickup_datetime`, `PULocationID`
- Aggregation: 263 taxi zones x 17,520 half-hour windows ≈ 4.6M records

---

## Methodology

### Feature Engineering
| Category | Features |
|----------|----------|
| Temporal | hour, minute_30, day_of_week, is_weekend, month, is_rush_hour, is_night |
| Cyclical | time_sin/cos, dow_sin/cos |
| Lag | Demand at t-1, t-2, t-3, t-6, t-12, t-24, t-48 |
| Rolling | Mean, std, max, min over 3h, 6h, 12h, 24h windows |
| Zone Stats | zone_mean, zone_std, zone_median, zone_hour_mean, zone_dow_mean |

### Models
1. **Linear Regression** — Simple baseline
2. **Random Forest** (200 trees, max_depth=20) — Non-linear patterns
3. **LSTM** (2-layer, 64 hidden, 24h sequence) — Temporal dependencies

### Failure Definition
- Failures = predictions with absolute error above the **85th percentile** (top 15%)
- Analyzed across temporal, spatial, and demand dimensions

### Explainability
- **SHAP TreeExplainer** for Random Forest
- **SHAP LinearExplainer** for Linear Regression
- Comparison of feature importance: normal vs failure cases

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Square Error |
| MAPE | Mean Absolute Percentage Error |

---

## Generated Outputs

### Visualizations (`outputs/figures/`)
- Demand overview, heatmap, and distribution plots
- Model comparison bar charts and scatter plots
- Failure rate by hour, day, month, and demand level
- Failure heatmaps (hour x day) per model
- Top failure zones bar charts
- SHAP summary, comparison, and dependence plots

### Reports (`outputs/reports/`)
- `model_metrics.csv` — Performance comparison table
- `zone_failures_*.csv` — Per-zone failure statistics
- `errors_*.parquet` — Full error records per model

---

## Configuration

All parameters are in `configs/config.yaml`. Key settings:

```yaml
data:
  time_window_minutes: 30    # Aggregation granularity
  year: 2019

models:
  test_size: 0.2             # Temporal split (last 20% = test)

failure_analysis:
  error_percentile: 85       # Top 15% errors = failures
```

---

## License

This project uses publicly available NYC TLC data. For academic use only.
