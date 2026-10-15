# Air Quality Index (AQI) Forecasting System

## 1. Project Summary & Objectives

Ambient air pollution is a critical public health crisis across metropolitan India. The **Air Quality Index (AQI)** serves as a composite index calculated from criteria atmospheric pollutants ($\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{CO}, \text{O}_3$). 

This final year engineering project develops a complete, production-grade Machine Learning system to forecast continuous AQI values using two foundational classical machine learning algorithms:
1. **Random Forest Regressor (RF)**: A bagging ensemble of $B=120$ decorrelated decision trees.
2. **Support Vector Regression (SVR)**: A structural risk minimization framework utilizing a Radial Basis Function (RBF) kernel in high-dimensional Hilbert space.



## 2. Benchmark Evaluation Results

The models were evaluated on an **out-of-sample chronological test partition** comprising $5,472$ daily observations.

| Metric | Random Forest Regressor | Support Vector Regression (SVR) | Optimal Direction |
| :--- | :---: | :---: | :---: |
| **Root Mean Squared Error (RMSE)** | **36.173** | **36.140** | Lower is better |
| **Mean Absolute Error (MAE)** | **20.210** | **20.384** | Lower is better |
| **Coefficient of Determination ($R^2$)** | **0.8549** | **0.8552** | Higher is better (Max 1.0) |
| **Input Feature Scaling** | Invariant (Unscaled) | Mandatory (`StandardScaler`) | Standardized ($\mu=0, \sigma=1$) |
| **Inference Latency** | $< 1.5\text{ ms}$ | $< 0.8\text{ ms}$ | Sub-millisecond |

### Random Forest Feature Importance Breakdown
- **$\text{PM}_{10}$ (Coarse Particulate Matter):** $37.43\%$
- **$\text{CO}$ (Carbon Monoxide):** $27.73\%$
- **$\text{PM}_{2.5}$ (Fine Particulate Matter):** $19.31\%$
- **$\text{NO}_2$ (Nitrogen Dioxide):** $12.66\%$
- **$\text{O}_3$ (Ground-Level Ozone):** $2.87\%$

---

## 3. Project Architecture & File Hierarchy

```text
sem7/ml/
│
├── city_day.csv                  # CPCB historical daily dataset (or any custom AQI dataset)
├── ml_pipeline.py                # Backend ML pipeline (data cleaning, scaling, training, evaluation)
├── app.py                        # Streamlit web application (2 focused tabs, no left panel navigation)
├── requirements.txt              # Production dependency specifications
├── README.md                     # Technical documentation and operational guide
│
└── models/                       # Serialized Scikit-learn artifacts
    ├── aqi_pipeline_bundle.joblib # Consolidated bundle (models, scaler, test results, metrics)
    ├── rf_model.joblib            # Standalone trained Random Forest Regressor
    ├── svr_model.joblib           # Standalone trained Support Vector Regressor
    ├── scaler.joblib              # Fitted StandardScaler (leakage-free)
    ├── evaluation_metrics.json    # Human-readable benchmark metrics
    └── test_predictions.csv       # Chronological test horizon actuals vs predictions
```

---

## 4. Key Engineering & Data Science Decisions

1. **City-Wise Time-Series Interpolation:**
   - Raw CPCB sensor readings have missing telemetry values due to station downtime or maintenance.
   - Simply dropping rows breaks temporal continuity. Imputing with zeros or global mean introduces severe bias.
   - Our pipeline groups records by **`City`** and executes **linear time-series interpolation** (`interpolate(method='linear', limit_direction='both')`), preserving localized geographic atmospheric trends without data contamination.
2. **Strict Chronological Train/Test Partitioning:**
   - Standard random shuffling (`train_test_split(shuffle=True)`) creates **lookahead bias (data leakage)** in time-series data.
   - The pipeline enforces strict chronological order: the first $80\%$ of historical data is used for training, and the subsequent $20\%$ is reserved strictly for out-of-sample validation.
3. **Dual Feature Standardization:**
   - SVR relies on Euclidean distance in kernel Hilbert space. Features with high absolute magnitudes (e.g., $\text{PM}_{10}$ ranging from $0-600$) would dominate small-magnitude features (e.g., $\text{CO}$ ranging from $0-5$).
   - A `StandardScaler` is fitted **strictly on the training partition** and then applied to test and live inference vectors.

---

## 5. Dataset Swapping: Will a New Dataset Work the Same?

**Yes!** If you bring a new dataset named `city_day.csv` (or replace the existing one), it will work seamlessly as long as:
1. **Required Columns Exist:** The dataset contains the criteria pollutants and target: `PM2.5`, `PM10`, `NO2`, `CO`, `O3`, and `AQI`. (The pipeline also accepts standard variations like `pm25`, `pm_25`, `pm10`, `datetime`, etc., and automatically supplies fallbacks if `City` or `Date` are not present).
2. **Models Must Be Retrained:** Replacing the CSV file on disk does not automatically overwrite pre-compiled models (`.joblib` files) until you retrain. You can retrain in one of two easy ways:
   - **From the Web App:** Open the top expander *"⚙️ Dataset & Model Synchronization"* and click **"🔄 Retrain Models Now"**.
   - **From the Terminal:** Run `python ml_pipeline.py`.

---

## 6. Quickstart & Execution Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the ML Pipeline (Backend Training)
```bash
python ml_pipeline.py
```

### Step 3: Launch the Streamlit Frontend Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 7. Streamlit Dashboard Features (2 Core Tabs)

1. **📊 Model Evaluation & Comparison:**
   - Side-by-side model architecture cards (Random Forest vs. SVR).
   - Comparative KPI metric cards for **RMSE**, **MAE**, and **$R^2$** with automatic winner indicators.
   - Grouped bar chart directly contrasting metrics.
   - Actual vs. Predicted time-series line chart tracking out-of-sample CPCB data.
   - Parity scatter plot ($y = x$) and Random Forest Feature Importance (MDI) breakdown.
2. **🔮 Real-Time AQI Forecasting Simulator:**
   - Quick atmospheric presets: *Clean Coastal Air*, *Typical Metro Day*, *Poor Urban Inversion*, *Severe Winter Smog*.
   - Interactive pollutant sliders for $\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{CO}, \text{O}_3$.
   - Real-time side-by-side forecasts from Random Forest and SVR.
   - Consensus Predicted AQI hero card with official CPCB health color category pill and advisory guidelines.
