import os
import glob
import json
import logging
import urllib.request
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# --------------------------------------------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("AQI_ML_Pipeline")

# --------------------------------------------------------------------------------------
# GLOBAL CONSTANTS & HYPERPARAMETERS
# --------------------------------------------------------------------------------------
# Key criteria air pollutants designated by CPCB (Central Pollution Control Board, India)
KEY_POLLUTANT_FEATURES = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3']
TARGET_VARIABLE = 'AQI'

# Chronological split proportion (80% historical training, 20% future testing)
TRAIN_TEST_SPLIT_RATIO = 0.80

# SVR hyperparameter tuning values optimized for standardized air quality metrics
SVR_HYPERPARAMS = {
    'kernel': 'rbf',       # Radial Basis Function kernel to capture complex non-linear pollutant interactions
    'C': 100.0,            # Regularization penalty parameter balancing margin violations vs. training error
    'epsilon': 0.1,        # Insensitivity margin tube where no penalty is associated in loss function
    'gamma': 'scale'       # Kernel coefficient 1 / (n_features * X.var())
}

# Random Forest hyperparameter tuning values
RF_HYPERPARAMS = {
    'n_estimators': 120,   # Number of unpruned decorrelated decision trees in the ensemble
    'max_depth': 16,       # Maximum tree depth to prevent memorization and overfitting
    'min_samples_split': 5,# Minimum samples required to split an internal node
    'random_state': 42,    # Reproducibility seed
    'n_jobs': -1           # Multi-threaded parallel processing
}

# Gradient Boosting hyperparameter tuning values
GBR_HYPERPARAMS = {
    'n_estimators': 120,   # Sequential boosting stages
    'learning_rate': 0.1,  # Shrinkage factor weighting contribution of each tree
    'max_depth': 5,        # Shallow depth trees for base learners to prevent overfitting
    'min_samples_split': 5,# Minimum samples for internal split
    'random_state': 42     # Reproducibility seed
}

# Artifacts output directory
MODELS_DIR = "models"


# --------------------------------------------------------------------------------------
# 1. DATA INGESTION & DISCOVERY
# --------------------------------------------------------------------------------------
def locate_or_fetch_dataset(custom_path: Optional[str] = None) -> str:
    """
    Locates the authentic CPCB 'city_day.csv' dataset in the working directory.
    If the file is not present, it automatically downloads the authentic Kaggle CPCB dataset.
    
    Academic Rationale:
    The Central Pollution Control Board (CPCB) publishes daily continuous ambient air
    quality monitoring data. Having an automated ingestion fallback ensures high reproducibility
    for examiners running this project across different machines without manual file management.
    
    Parameters:
        custom_path (str, optional): Custom path provided by the user or Streamlit dashboard.
        
    Returns:
        str: Validated file path to city_day.csv.
    """
    # 1. Check if user specified a valid custom path
    if custom_path and os.path.exists(custom_path):
        logger.info(f"Using explicitly specified dataset path: {custom_path}")
        return custom_path

    # 2. Check current working directory for city_day.csv or variants
    candidates = [
        "city_day.csv",
        "Air Quality Data in India (2015 - 2024)/city_day.csv",
        "data/city_day.csv",
        "../city_day.csv"
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            logger.info(f"Found local dataset at: {os.path.abspath(candidate)}")
            return candidate

    # 3. Search via glob for any matching CSV in directory
    matched_files = glob.glob("*city_day*.csv") + glob.glob("*/*city_day*.csv")
    if matched_files:
        logger.info(f"Discovered matching dataset: {matched_files[0]}")
        return matched_files[0]

    # 4. If completely missing, download the authenticated CPCB open-access mirror
    download_url = "https://raw.githubusercontent.com/Sudhanshu-Ambastha/Air-Quality-Prediction-Bot/master/city_day.csv"
    target_file = "city_day.csv"
    logger.warning(f"Dataset not detected locally. Fetching authenticated CPCB dataset from {download_url}...")
    try:
        urllib.request.urlretrieve(download_url, target_file)
        logger.info(f"Successfully downloaded CPCB dataset to '{target_file}' ({os.path.getsize(target_file)} bytes).")
        return target_file
    except Exception as exc:
        raise FileNotFoundError(
            f"Failed to automatically retrieve dataset. Please place 'city_day.csv' into {os.getcwd()}. "
            f"Original Error: {exc}"
        )


# --------------------------------------------------------------------------------------
# 2. DATA LOADING & TIME-SERIES PREPROCESSING
# --------------------------------------------------------------------------------------
def load_and_preprocess_data(
    filepath: str = "city_day.csv",
    city_filter: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw CPCB air quality measurements and performs time-series-aware preprocessing.
    
    ACADEMIC RATIONALE FOR MISSING VALUE TREATMENT:
    Air quality measurements from monitoring stations frequently suffer from missing sensor values
    due to telemetry dropouts, sensor calibration cycles, or power outages.
    - Dropping rows: Discards valuable historical context and breaks the time series continuity.
    - Imputing with zeros: Catastrophically distorts statistics, falsely implying zero pollution.
    - Imputing with global mean: Introduces temporal leakage and flattens seasonal smog spikes.
    - TIME-SERIES INTERPOLATION (Linear / Time): Models the continuous physical dispersion of
      atmospheric particulates by estimating intermediate values between surrounding timestamps.
      This interpolation is executed STRICTLY PER CITY to prevent geographic pollution leakage.
      
    Parameters:
        filepath (str): Path to CSV data file.
        city_filter (str, optional): If specified (e.g. 'Delhi'), filters preprocessing to that city.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and metadata diagnostics dictionary.
    """
    valid_path = locate_or_fetch_dataset(filepath)
    logger.info(f"Loading raw dataset from '{valid_path}'...")
    raw_df = pd.read_csv(valid_path)

    # Clean whitespace from column names
    raw_df.columns = [str(c).strip() for c in raw_df.columns]

    # Flexible column mapping dictionary for varying dataset schemas
    col_alias_map = {
        'pm2.5': 'PM2.5', 'pm25': 'PM2.5', 'pm_25': 'PM2.5', 'pm2_5': 'PM2.5',
        'pm10': 'PM10', 'pm_10': 'PM10',
        'no2': 'NO2', 'nitrogen_dioxide': 'NO2',
        'co': 'CO', 'carbon_monoxide': 'CO',
        'o3': 'O3', 'ozone': 'O3',
        'aqi': 'AQI', 'air_quality_index': 'AQI',
        'date': 'Date', 'datetime': 'Date', 'timestamp': 'Date',
        'city': 'City', 'station': 'City', 'location': 'City'
    }

    # Map columns case-insensitively if exact names are not present
    rename_dict = {}
    for col in raw_df.columns:
        col_lower = col.lower()
        if col_lower in col_alias_map and col_alias_map[col_lower] not in raw_df.columns:
            rename_dict[col] = col_alias_map[col_lower]
    if rename_dict:
        raw_df = raw_df.rename(columns=rename_dict)
        logger.info(f"Mapped dataset columns to canonical names: {rename_dict}")

    # Fallback for City if missing
    if 'City' not in raw_df.columns:
        raw_df['City'] = 'Station_1'
        logger.info("Column 'City' not found; defaulted to 'Station_1'")

    # Fallback for Date if missing
    if 'Date' not in raw_df.columns:
        raw_df['Date'] = pd.date_range(end=pd.Timestamp.now(), periods=len(raw_df), freq='D')
        logger.info("Column 'Date' not found; generated sequential daily timestamps.")

    # Validate essential criteria pollutant columns and target
    required_cols = ['City', 'Date'] + KEY_POLLUTANT_FEATURES + [TARGET_VARIABLE]
    missing_cols = [c for c in required_cols if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required pollutant columns: {missing_cols}. "
            f"Present columns: {list(raw_df.columns)}. "
            f"Ensure the dataset has: PM2.5, PM10, NO2, CO, O3, and AQI."
        )

    # Parse temporal index safely
    raw_df['Date'] = pd.to_datetime(raw_df['Date'], errors='coerce')
    raw_df = raw_df.dropna(subset=['Date'])
    raw_df = raw_df.sort_values(by=['City', 'Date']).reset_index(drop=True)

    # Initial null statistics for transparency and academic reporting
    initial_nulls = raw_df[KEY_POLLUTANT_FEATURES + [TARGET_VARIABLE]].isnull().sum().to_dict()
    total_raw_rows = len(raw_df)

    # Filter by specific city if requested
    if city_filter and city_filter.strip().lower() != "all":
        raw_df = raw_df[raw_df['City'].astype(str).str.strip().str.lower() == city_filter.strip().lower()].copy()
        if raw_df.empty:
            raise ValueError(f"City '{city_filter}' not found in dataset. Available: {list(raw_df['City'].unique())}")
        logger.info(f"Filtered analysis strictly to city: '{city_filter}' ({len(raw_df)} records).")

    # Group by City and apply time-series linear interpolation
    # This ensures Delhi's high winter smog values never interpolate into Bengaluru's cleaner air!
    cleaned_group_list = []
    interpolation_cols = KEY_POLLUTANT_FEATURES + [TARGET_VARIABLE]

    for city_name, group in raw_df.groupby('City', as_index=False):
        group = group.sort_values('Date').copy()
        
        # 1. Linear interpolation along the temporal index
        group[interpolation_cols] = group[interpolation_cols].interpolate(
            method='linear',
            limit_direction='both'
        )
        
        # 2. Backward-fill and forward-fill for any persistent edge boundary cases
        group[interpolation_cols] = group[interpolation_cols].bfill().ffill()
        
        cleaned_group_list.append(group)

    clean_df = pd.concat(cleaned_group_list, ignore_index=True)

    # Drop any remaining unfillable rows (e.g. if an entire city had 0 valid records for a pollutant)
    clean_df = clean_df.dropna(subset=interpolation_cols).reset_index(drop=True)
    clean_df = clean_df.sort_values('Date').reset_index(drop=True)

    post_nulls = clean_df[KEY_POLLUTANT_FEATURES + [TARGET_VARIABLE]].isnull().sum().to_dict()
    logger.info(f"Preprocessing completed. Initial rows: {total_raw_rows} -> Cleaned rows: {len(clean_df)}")

    diagnostics = {
        'total_raw_records': total_raw_rows,
        'cleaned_records': len(clean_df),
        'initial_null_counts': initial_nulls,
        'post_null_counts': post_nulls,
        'date_min': str(clean_df['Date'].min().date()),
        'date_max': str(clean_df['Date'].max().date()),
        'unique_cities': sorted(clean_df['City'].unique().tolist())
    }

    return clean_df, diagnostics


# --------------------------------------------------------------------------------------
# 3. FEATURE ENGINEERING & STRICT CHRONOLOGICAL SPLITTING
# --------------------------------------------------------------------------------------
def prepare_train_test_split(
    df: pd.DataFrame,
    features: list = KEY_POLLUTANT_FEATURES,
    target: str = TARGET_VARIABLE,
    split_ratio: float = TRAIN_TEST_SPLIT_RATIO
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Performs a strict chronological Train/Test split without shuffling.
    
    ACADEMIC RATIONALE (NO RANDOM SHUFFLING):
    In standard cross-sectional ML (e.g., image or tabular fraud classification), random train_test_split
    is standard practice. HOWEVER, IN TIME-SERIES AIR QUALITY FORECASTING:
    - Random shuffling introduces FATAL LOOKAHEAD BIAS (data leakage), where future pollution levels
      leak into the training partition, artificially inflating test scores.
    - In real-world deployment, an environmental model trained on past observations must forecast
      unseen future air quality.
    - Therefore, the dataset is sorted strictly by Date, and the first 80% is allocated for training,
      while the final 20% chronological segment is reserved strictly for out-of-sample evaluation.
    
    Returns:
        X_train, X_test, y_train, y_test, train_dates, test_dates
    """
    # Sort strictly chronologically
    df_sorted = df.sort_values(by='Date').reset_index(drop=True)
    
    split_index = int(len(df_sorted) * split_ratio)
    
    train_slice = df_sorted.iloc[:split_index].copy()
    test_slice = df_sorted.iloc[split_index:].copy()
    
    X_train = train_slice[features].copy()
    y_train = train_slice[target].copy()
    train_dates = train_slice['Date'].copy()
    
    X_test = test_slice[features].copy()
    y_test = test_slice[target].copy()
    test_dates = test_slice['Date'].copy()
    
    logger.info(
        f"Chronological Split Complete: "
        f"Training set: {len(X_train)} samples ({train_dates.min().date()} to {train_dates.max().date()}) | "
        f"Testing set: {len(X_test)} samples ({test_dates.min().date()} to {test_dates.max().date()})"
    )
    
    return X_train, X_test, y_train, y_test, train_dates, test_dates


# --------------------------------------------------------------------------------------
# 4. FEATURE STANDARDIZATION
# --------------------------------------------------------------------------------------
def apply_feature_scaling(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fits a StandardScaler strictly on the training partition and transforms train and test partitions.
    
    ACADEMIC RATIONALE:
    1. SVR Dependency: Support Vector Regression calculates margins and distances using kernel inner
       products (e.g. ||x_i - x_j||^2 in RBF kernel). If PM10 has a range of 0-800 while CO has a
       range of 0-5, unscaled SVR will be completely dominated by PM10, rendering CO statistically invisible.
    2. Random Forest Invariance: Random Forest is an ensemble of axis-aligned orthogonal decision trees
       and is invariant to monotonic feature scaling; however, scaled features do not harm RF.
    3. Leakage Prevention: Scaler parameters (mean mu and standard deviation sigma) MUST be calculated
       strictly on X_train, then applied to X_test to simulate an authentic out-of-sample environment.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("StandardScaler successfully fitted on X_train and applied to X_train & X_test.")
    return X_train_scaled, X_test_scaled, scaler


# --------------------------------------------------------------------------------------
# 5. MODEL TRAINING: RANDOM FOREST & SUPPORT VECTOR REGRESSION
# --------------------------------------------------------------------------------------
def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparams: Dict[str, Any] = RF_HYPERPARAMS
) -> RandomForestRegressor:
    """
    Trains a Scikit-Learn RandomForestRegressor.
    
    Theory:
    Random Forest is an ensemble meta-estimator that fits multiple decorrelated decision trees
    on various sub-samples of the dataset and uses averaging to improve predictive accuracy
    and control over-fitting. It implicitly computes feature importances via MDI (Mean Decrease Impurity).
    """
    logger.info(f"Training Random Forest Regressor with params: {hyperparams}...")
    rf_model = RandomForestRegressor(**hyperparams)
    rf_model.fit(X_train, y_train)
    logger.info("Random Forest Regressor training completed successfully.")
    return rf_model


def train_support_vector_regression(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    hyperparams: Dict[str, Any] = SVR_HYPERPARAMS
) -> SVR:
    """
    Trains a Scikit-Learn Support Vector Regressor (SVR).
    
    Theory:
    SVR solves the optimization problem:
        min (1/2)||w||^2 + C * sum(xi_i + xi_i^*)
    subject to:
        |y_i - <w, phi(x_i)> - b| <= epsilon
    The RBF kernel projects features into an infinite-dimensional Hilbert space, enabling
    the algorithm to find an optimal maximum-margin hyperplane for continuous AQI prediction.
    """
    logger.info(f"Training Support Vector Regressor (SVR) with params: {hyperparams}...")
    svr_model = SVR(**hyperparams)
    svr_model.fit(X_train_scaled, y_train)
    logger.info("Support Vector Regressor (SVR) training completed successfully.")
    return svr_model


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparams: Dict[str, Any] = GBR_HYPERPARAMS
) -> GradientBoostingRegressor:
    """
    Trains a Scikit-Learn GradientBoostingRegressor.
    
    Theory:
    Gradient Boosting constructs an additive model in a forward stage-wise manner.
    At each iteration, a regression tree is fitted on the negative gradient (pseudo-residuals)
    of the loss function, enabling rapid convergence and powerful non-linear modeling.
    """
    logger.info(f"Training Gradient Boosting Regressor with params: {hyperparams}...")
    gbr_model = GradientBoostingRegressor(**hyperparams)
    gbr_model.fit(X_train, y_train)
    logger.info("Gradient Boosting Regressor training completed successfully.")
    return gbr_model


# --------------------------------------------------------------------------------------
# 6. EVALUATION METRICS CALCULATION
# --------------------------------------------------------------------------------------
def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str
) -> Dict[str, float]:
    """
    Computes standard regression benchmark metrics:
    - Root Mean Squared Error (RMSE): Penalizes larger prediction outliers heavily.
    - Mean Absolute Error (MAE): Measures average magnitude of errors in identical units (AQI points).
    - R^2 Score (Coefficient of Determination): Proportion of variance explained by model (max 1.0).
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    logger.info(f"[{model_name}] Evaluation Results -> RMSE: {rmse:.3f} | MAE: {mae:.3f} | R^2: {r2:.4f}")
    return {
        'model_name': model_name,
        'rmse': round(rmse, 3),
        'mae': round(mae, 3),
        'r2_score': round(r2, 4)
    }


# --------------------------------------------------------------------------------------
# 7. ARTIFACT PERSISTENCE
# --------------------------------------------------------------------------------------
def serialize_artifacts(
    rf_model: RandomForestRegressor,
    svr_model: SVR,
    gbr_model: GradientBoostingRegressor,
    scaler: StandardScaler,
    metrics: Dict[str, Any],
    feature_importances: Dict[str, Any],
    test_results_df: pd.DataFrame,
    diagnostics: Dict[str, Any],
    output_dir: str = MODELS_DIR
) -> str:
    """
    Persists all trained models, preprocessors, evaluation metrics, and test results
    into binary and JSON formats for seamless low-latency consumption by the web frontend.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual components for flexibility
    rf_path = os.path.join(output_dir, "rf_model.joblib")
    svr_path = os.path.join(output_dir, "svr_model.joblib")
    gbr_path = os.path.join(output_dir, "gbr_model.joblib")
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    bundle_path = os.path.join(output_dir, "aqi_pipeline_bundle.joblib")
    metrics_json_path = os.path.join(output_dir, "evaluation_metrics.json")
    test_results_path = os.path.join(output_dir, "test_predictions.csv")
    
    joblib.dump(rf_model, rf_path)
    joblib.dump(svr_model, svr_path)
    joblib.dump(gbr_model, gbr_path)
    joblib.dump(scaler, scaler_path)
    
    # Consolidated package for web frontend
    artifacts_bundle = {
        'rf_model': rf_model,
        'svr_model': svr_model,
        'gbr_model': gbr_model,
        'scaler': scaler,
        'features': KEY_POLLUTANT_FEATURES,
        'target': TARGET_VARIABLE,
        'metrics': metrics,
        'feature_importances': feature_importances,
        'diagnostics': diagnostics,
        'test_results': test_results_df
    }
    joblib.dump(artifacts_bundle, bundle_path)
    
    # Save human-readable JSON metrics for examiners/reviewers
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
        
    test_results_df.to_csv(test_results_path, index=False)
    
    logger.info(f"All model artifacts successfully serialized into directory: '{os.path.abspath(output_dir)}'")
    return bundle_path


# --------------------------------------------------------------------------------------
# 8. MASTER PIPELINE ORCHESTRATOR
# --------------------------------------------------------------------------------------
def execute_pipeline(
    filepath: str = "city_day.csv",
    city_filter: Optional[str] = None,
    output_dir: str = MODELS_DIR
) -> Dict[str, Any]:
    """
    Orchestrates the complete end-to-end Machine Learning pipeline from raw CSV to serialized models.
    Trains and evaluates 3 classical algorithms:
      1. Random Forest Regressor (Bagging Ensemble)
      2. Support Vector Regression (Kernel Method)
      3. Gradient Boosting Regressor (Boosting Ensemble)
    """
    logger.info("=" * 70)
    logger.info("STARTING AIR QUALITY INDEX (AQI) FORECASTING ML PIPELINE")
    logger.info("=" * 70)
    
    # Step 1: Preprocessing & Time-Series Interpolation
    clean_df, diagnostics = load_and_preprocess_data(filepath, city_filter)
    
    # Step 2: Chronological Splitting
    X_train, X_test, y_train, y_test, train_dates, test_dates = prepare_train_test_split(
        clean_df,
        features=KEY_POLLUTANT_FEATURES,
        target=TARGET_VARIABLE,
        split_ratio=TRAIN_TEST_SPLIT_RATIO
    )
    
    # Step 3: Feature Scaling (Strictly on X_train)
    X_train_scaled, X_test_scaled, scaler = apply_feature_scaling(X_train, X_test)
    
    # Step 4: Train Random Forest Regressor (Bagging)
    rf_model = train_random_forest(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_metrics = compute_regression_metrics(y_test.values, y_pred_rf, "Random Forest Regressor")
    
    # Step 5: Train Support Vector Regressor (SVR - Kernel)
    svr_model = train_support_vector_regression(X_train_scaled, y_train)
    y_pred_svr = svr_model.predict(X_test_scaled)
    svr_metrics = compute_regression_metrics(y_test.values, y_pred_svr, "Support Vector Regression (SVR)")
    
    # Step 6: Train Gradient Boosting Regressor (GBR - Boosting)
    gbr_model = train_gradient_boosting(X_train, y_train)
    y_pred_gbr = gbr_model.predict(X_test)
    gbr_metrics = compute_regression_metrics(y_test.values, y_pred_gbr, "Gradient Boosting Regressor")
    
    # Extract Feature Importances from RF and GBR
    rf_importances = dict(zip(KEY_POLLUTANT_FEATURES, rf_model.feature_importances_.round(4)))
    gbr_importances = dict(zip(KEY_POLLUTANT_FEATURES, gbr_model.feature_importances_.round(4)))
    feature_importances = {
        'Random_Forest': rf_importances,
        'Gradient_Boosting': gbr_importances,
        **rf_importances
    }
    
    # Step 7: Construct Out-of-Sample Test Evaluation DataFrame
    test_results_df = pd.DataFrame({
        'Date': test_dates.values,
        'Actual_AQI': y_test.values,
        'RF_Predicted_AQI': np.round(y_pred_rf, 2),
        'SVR_Predicted_AQI': np.round(y_pred_svr, 2),
        'GBR_Predicted_AQI': np.round(y_pred_gbr, 2),
        'RF_Absolute_Error': np.round(np.abs(y_test.values - y_pred_rf), 2),
        'SVR_Absolute_Error': np.round(np.abs(y_test.values - y_pred_svr), 2),
        'GBR_Absolute_Error': np.round(np.abs(y_test.values - y_pred_gbr), 2)
    })
    
    all_metrics = {
        'Random_Forest': rf_metrics,
        'Support_Vector_Regression': svr_metrics,
        'Gradient_Boosting': gbr_metrics,
        'Train_Samples': len(X_train),
        'Test_Samples': len(X_test),
        'Split_Ratio': f"{int(TRAIN_TEST_SPLIT_RATIO*100)}/{int((1-TRAIN_TEST_SPLIT_RATIO)*100)} Chronological",
        'Feature_Importances': feature_importances
    }
    
    # Step 8: Serialize Artifacts
    bundle_path = serialize_artifacts(
        rf_model=rf_model,
        svr_model=svr_model,
        gbr_model=gbr_model,
        scaler=scaler,
        metrics=all_metrics,
        feature_importances=feature_importances,
        test_results_df=test_results_df,
        diagnostics=diagnostics,
        output_dir=output_dir
    )
    
    logger.info("=" * 70)
    logger.info("AQI ML PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info(f"Random Forest     -> RMSE: {rf_metrics['rmse']} | MAE: {rf_metrics['mae']} | R^2: {rf_metrics['r2_score']}")
    logger.info(f"SVR Model         -> RMSE: {svr_metrics['rmse']} | MAE: {svr_metrics['mae']} | R^2: {svr_metrics['r2_score']}")
    logger.info(f"Gradient Boosting -> RMSE: {gbr_metrics['rmse']} | MAE: {gbr_metrics['mae']} | R^2: {gbr_metrics['r2_score']}")
    logger.info("=" * 70)
    
    return {
        'bundle_path': bundle_path,
        'metrics': all_metrics,
        'clean_df': clean_df,
        'test_results': test_results_df
    }


if __name__ == "__main__":
    # When executed as a standalone script, runs the pipeline on the authentic CPCB dataset
    execute_pipeline()
