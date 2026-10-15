# AQI Forecasting System

This project is a practical air-quality prediction system built for Indian cities. It uses historical pollutant data to estimate the Air Quality Index (AQI) and gives a simple way to compare model performance and generate forecasts.

The focus is not just on building a model, but on making it usable: clean the data, train the models, compare results, and visualise the output in a dashboard.

## Why this project matters

Air pollution is a growing problem in many cities, and AQI is one of the easiest ways to understand how healthy or unhealthy the air is. This project looks at common pollutants such as PM2.5, PM10, NO2, CO, and O3, and tries to estimate AQI based on those values.

## What it does

- Cleans and prepares AQI data for modelling
- Handles missing values by city-wise interpolation
- Splits the dataset in a time-aware way so the model is evaluated fairly
- Trains machine learning models to predict AQI
- Compares model accuracy using common regression metrics
- Provides a dashboard for exploring results and making forecasts

## Models used

The project compares the following models:

- Random Forest Regressor
- Support Vector Regression (SVR)

These are trained on historical city-level AQI data and evaluated on unseen test data.

## Evaluation snapshot

The models were checked on a chronological test set with 5,472 daily observations.

| Metric | Random Forest | SVR | Better when |
| :--- | :---: | :---: | :---: |
| RMSE | 36.173 | 36.140 | Lower |
| MAE | 20.210 | 20.384 | Lower |
| R² | 0.8549 | 0.8552 | Higher |

In simple terms, both models perform well, and the SVR gives a slightly better R² score. The Random Forest remains very competitive and often gives a good balance between accuracy and interpretability.

### Feature importance (Random Forest)

The model gives the most weight to these pollution indicators:

- PM10: 37.43%
- CO: 27.73%
- PM2.5: 19.31%
- NO2: 12.66%
- O3: 2.87%

This highlights that particulate matter and carbon monoxide are especially important in predicting AQI.

## Project structure

```text
AQI/
├── city_day.csv                  # AQI dataset
├── ml_pipeline.py                # Data cleaning, training, evaluation pipeline
├── server.py                     # Flask backend for the dashboard
├── app.py                        # Streamlit version (alternative interface)
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── static/                       # Frontend assets
│   ├── index.html
│   ├── css/
│   └── js/
└── models/                       # Trained model files and evaluation outputs
    ├── aqi_pipeline_bundle.joblib
    ├── rf_model.joblib
    ├── svr_model.joblib
    ├── scaler.joblib
    ├── evaluation_metrics.json
    └── test_predictions.csv
```

## Key decisions in the pipeline

1. City-wise interpolation
   - Missing sensor readings are filled in within each city rather than using a single global average.
   - This keeps the time pattern more realistic and avoids introducing misleading values.

2. Chronological train/test split
   - The model is trained on past data and tested on later data.
   - This is important because random shuffling would leak future information into the training process.

3. Feature scaling for SVR
   - SVR works better when features are on a similar scale.
   - The scaler is fitted only on training data and then applied to test and real-time inputs.

## Using a different dataset

The project is designed to work with a CSV file named `city_day.csv`. If you replace it with another AQI dataset, it can still work as long as the file contains the expected pollutant columns such as:

- PM2.5
- PM10
- NO2
- CO
- O3
- AQI

If the new dataset has slightly different column names, the preprocessing code is flexible enough to handle common variations. However, you should retrain the models after updating the dataset.

## Getting started

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Train the models

```bash
python ml_pipeline.py
```

### 3) Start the web app

```bash
python server.py
```

Then open:

```text
http://localhost:5000
```

If you want to use the Streamlit version instead, you can also run:

```bash
streamlit run app.py
```

## What the dashboard shows

The web app includes:

- Model comparison cards
- Training and testing metrics
- Time-series prediction plots
- Forecasting inputs for pollutant levels
- AQI health category and advisory messages

This makes it easier to see not just the numbers, but also what the model is predicting in real time.

## Notes

This is a small but useful end-to-end machine learning project for AQI forecasting. It can be used as a learning project, a demo for academic work, or a starting point for a larger air-quality monitoring system.

If you want, I can also give this README a cleaner project-style layout with a featured screenshot section, installation notes, and a more polished final presentation version.
