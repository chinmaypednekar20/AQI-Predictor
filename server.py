import os
import json
import logging
from typing import Dict, Any, Optional

from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import numpy as np
import joblib

from ml_pipeline import (
    KEY_POLLUTANT_FEATURES,
    TARGET_VARIABLE,
    MODELS_DIR,
    locate_or_fetch_dataset,
    load_and_preprocess_data,
    execute_pipeline
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("AQI_Server")

app = Flask(__name__, static_folder="static")

# --------------------------------------------------------------------------------------
# GLOBAL CACHED ARTIFACTS
# --------------------------------------------------------------------------------------
pipeline_cache: Dict[str, Any] = {}


def load_artifacts(force_retrain: bool = False) -> Dict[str, Any]:
    """Loads model artifacts from disk or executes training pipeline if not present."""
    global pipeline_cache
    bundle_path = os.path.join(MODELS_DIR, "aqi_pipeline_bundle.joblib")

    if force_retrain or not os.path.exists(bundle_path):
        logger.info("Executing ML training pipeline...")
        execute_pipeline(output_dir=MODELS_DIR)

    logger.info("Loading cached model bundle into memory...")
    bundle = joblib.load(bundle_path)
    
    # Ensure diagnostics and dataset info are available
    if 'diagnostics' not in bundle:
        _, diagnostics = load_and_preprocess_data()
        bundle['diagnostics'] = diagnostics

    pipeline_cache = bundle
    return bundle


def get_cached_bundle() -> Dict[str, Any]:
    global pipeline_cache
    if not pipeline_cache:
        pipeline_cache = load_artifacts()
    return pipeline_cache


# --------------------------------------------------------------------------------------
# CPCB NAQI HEALTH CLASSIFICATION LOGIC
# --------------------------------------------------------------------------------------
def classify_cpcb_aqi(aqi_val: float) -> Dict[str, str]:
    """Classifies continuous AQI into official Central Pollution Control Board (CPCB) categories."""
    aqi_round = round(aqi_val, 1)
    if aqi_round <= 50:
        return {
            'category': 'Good',
            'range': '0 - 50',
            'color': '#10B981',        # Emerald green
            'bg_light': 'rgba(16, 185, 129, 0.15)',
            'border': '#059669',
            'advisory': 'Minimal health impact. Air quality is exceptionally clean and satisfactory for all outdoor activities.'
        }
    elif aqi_round <= 100:
        return {
            'category': 'Satisfactory',
            'range': '51 - 100',
            'color': '#84CC16',        # Lime green
            'bg_light': 'rgba(132, 204, 22, 0.15)',
            'border': '#65A30D',
            'advisory': 'Minor breathing discomfort to highly sensitive individuals. Acceptable for the general public.'
        }
    elif aqi_round <= 200:
        return {
            'category': 'Moderate',
            'range': '101 - 200',
            'color': '#F59E0B',        # Amber yellow
            'bg_light': 'rgba(245, 158, 11, 0.15)',
            'border': '#D97706',
            'advisory': 'Breathing discomfort to people with lung disease, asthma, and vulnerable individuals such as children.'
        }
    elif aqi_round <= 300:
        return {
            'category': 'Poor',
            'range': '201 - 300',
            'color': '#F97316',        # Orange
            'bg_light': 'rgba(249, 115, 22, 0.15)',
            'border': '#EA580C',
            'advisory': 'Breathing discomfort to most people on prolonged outdoor exposure. Limit heavy exertion.'
        }
    elif aqi_round <= 400:
        return {
            'category': 'Very Poor',
            'range': '301 - 400',
            'color': '#EF4444',        # Red
            'bg_light': 'rgba(239, 68, 68, 0.15)',
            'border': '#DC2626',
            'advisory': 'Respiratory illness on prolonged exposure. Significant distress for cardiac and asthma patients.'
        }
    else:
        return {
            'category': 'Severe',
            'range': '401 - 500+',
            'color': '#991B1B',        # Deep Crimson / Burgundy
            'bg_light': 'rgba(153, 27, 27, 0.25)',
            'border': '#7F1D1D',
            'advisory': 'Severe health alert! May cause respiratory impact even in healthy people; serious risk for vulnerable populations.'
        }


# --------------------------------------------------------------------------------------
# API ROUTES
# --------------------------------------------------------------------------------------
@app.route("/")
def index():
    """Serves the frontend single-page dashboard."""
    return send_from_directory("static", "index.html")


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Returns evaluation metrics, diagnostics, feature importances, and chronological predictions."""
    try:
        bundle = get_cached_bundle()
        metrics = bundle['metrics']
        diagnostics = bundle.get('diagnostics', {})
        test_df = bundle['test_results']
        
        # Sample test results for ultra-responsive time-series charting (approx 350-400 evenly spaced points)
        step = max(1, len(test_df) // 350)
        sampled_test = test_df.iloc[::step].copy()
        
        # Format timestamps cleanly
        chart_dates = [str(d)[:10] for d in sampled_test['Date']]
        chart_actual = [round(float(v), 1) if pd.notnull(v) else None for v in sampled_test['Actual_AQI']]
        chart_rf = [round(float(v), 1) if pd.notnull(v) else None for v in sampled_test['RF_Predicted_AQI']]
        chart_svr = [round(float(v), 1) if pd.notnull(v) else None for v in sampled_test['SVR_Predicted_AQI']]
        chart_gbr = [round(float(v), 1) if pd.notnull(v) else None for v in sampled_test.get('GBR_Predicted_AQI', sampled_test['RF_Predicted_AQI'])]

        feature_importances = bundle.get('feature_importances', {})
        rf_fi = feature_importances.get('Random_Forest', {})
        gbr_fi = feature_importances.get('Gradient_Boosting', {})

        return jsonify({
            'status': 'success',
            'metrics': metrics,
            'feature_importances': {
                'Random_Forest': rf_fi,
                'Gradient_Boosting': gbr_fi
            },
            'diagnostics': {
                'total_records': diagnostics.get('cleaned_records', len(test_df) * 5),
                'date_min': diagnostics.get('date_min', '2015-01-01'),
                'date_max': diagnostics.get('date_max', '2020-07-01'),
                'train_samples': metrics.get('Train_Samples', 21888),
                'test_samples': metrics.get('Test_Samples', 5472),
                'split_ratio': metrics.get('Split_Ratio', '80/20 Chronological')
            },
            'timeseries': {
                'dates': chart_dates,
                'actual': chart_actual,
                'rf': chart_rf,
                'svr': chart_svr,
                'gbr': chart_gbr
            }
        })
    except Exception as exc:
        logger.error(f"Error fetching metrics: {exc}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route("/api/predict", methods=["POST"])
def predict_aqi():
    """
    Accepts 5 criteria pollutant levels, predicts AQI using all 3 algorithms,
    and returns individual forecasts + consensus + CPCB advisory.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'status': 'error', 'message': 'Missing JSON payload.'}), 400

        # Extract features with sensible defaults
        pm25 = float(data.get('PM2.5', 55.0))
        pm10 = float(data.get('PM10', 110.0))
        no2 = float(data.get('NO2', 38.0))
        co = float(data.get('CO', 1.2))
        o3 = float(data.get('O3', 42.0))

        bundle = get_cached_bundle()
        rf_model = bundle['rf_model']
        svr_model = bundle['svr_model']
        gbr_model = bundle['gbr_model']
        scaler = bundle['scaler']

        # Construct DataFrame feature vector to preserve feature names for tree models
        features_df = pd.DataFrame([{
            'PM2.5': pm25,
            'PM10': pm10,
            'NO2': no2,
            'CO': co,
            'O3': o3
        }])

        # Scale features strictly for SVR
        features_scaled = scaler.transform(features_df)

        # Generate predictions
        pred_rf = max(0.0, float(rf_model.predict(features_df)[0]))
        pred_svr = max(0.0, float(svr_model.predict(features_scaled)[0]))
        pred_gbr = max(0.0, float(gbr_model.predict(features_df)[0]))

        # Consensus ensemble average
        consensus = round((pred_rf + pred_svr + pred_gbr) / 3.0, 1)
        cpcb = classify_cpcb_aqi(consensus)

        return jsonify({
            'status': 'success',
            'predictions': {
                'rf': round(pred_rf, 1),
                'svr': round(pred_svr, 1),
                'gbr': round(pred_gbr, 1),
                'consensus': consensus
            },
            'cpcb': cpcb,
            'inputs': {
                'PM2.5': pm25,
                'PM10': pm10,
                'NO2': no2,
                'CO': co,
                'O3': o3
            }
        })
    except Exception as exc:
        logger.error(f"Error during prediction: {exc}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route("/api/retrain", methods=["POST"])
def retrain_models():
    """Retrains all models on the latest dataset and updates cached bundle."""
    try:
        logger.info("Retraining models requested via API...")
        load_artifacts(force_retrain=True)
        return jsonify({
            'status': 'success',
            'message': 'Models successfully retrained and cached in memory!'
        })
    except Exception as exc:
        logger.error(f"Error during retraining: {exc}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(exc)}), 500


# --------------------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    load_artifacts()
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting AQI Forecasting Server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
