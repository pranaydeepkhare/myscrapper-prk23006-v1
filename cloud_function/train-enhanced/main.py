# main.py - ENHANCED TRAINING WITH RANDOM FOREST & INTERPRETABILITY
# Location: cloud_function/train-enhanced/main.py
# Features:
# 1. Hyperparameter tuning with Optuna
# 2. Random Forest Regressor (DEFAULT MODEL)
# 3. Uses ALL 10 features including the 4 new ones (color, city, state, zip_code)
# 4. Generates interpretability outputs:
#    - Permutation importance (all features)
#    - Partial Dependence Plots (top 3 features)
# 5. Saves predictions + interpretability to GCS

import os
import io
import json
import logging
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from google.cloud import storage

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for GCS
import matplotlib.pyplot as plt

# Optuna for hyperparameter tuning
import optuna
from optuna.samplers import TPESampler

# ---- ENV ----
PROJECT_ID     = os.getenv("PROJECT_ID", "")
GCS_BUCKET     = os.getenv("GCS_BUCKET", "")
DATA_KEY       = os.getenv("DATA_KEY", "structured/datasets/listings_master_llm_enhanced.csv")
OUTPUT_PREFIX  = os.getenv("OUTPUT_PREFIX", "enhanced_results")
TIMEZONE       = os.getenv("TIMEZONE", "America/New_York")
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")
N_TRIALS       = int(os.getenv("N_TRIALS", "20"))  # Optuna trials
MODEL_TYPE     = os.getenv("MODEL_TYPE", "random_forest")  # DEFAULT: random_forest

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")

def _read_csv_from_gcs(client: storage.Client, bucket: str, key: str) -> pd.DataFrame:
    b = client.bucket(bucket)
    blob = b.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))

def _write_csv_to_gcs(client: storage.Client, bucket: str, key: str, df: pd.DataFrame):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")

def _write_bytes_to_gcs(client: storage.Client, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream"):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(data, content_type=content_type)

def _clean_numeric(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    mask = y_true.notna()
    if not mask.any():
        return {}
    
    y_t = y_true[mask]
    y_p = y_pred[mask]
    
    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    
    # MAPE (avoid division by zero)
    mape_mask = y_t != 0
    if mape_mask.any():
        mape = mean_absolute_percentage_error(y_t[mape_mask], y_p[mape_mask]) * 100
    else:
        mape = None
    
    # Bias (mean error)
    bias = np.mean(y_p - y_t)
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape) if mape is not None else None,
        "bias": float(bias),
        "n_samples": int(mask.sum())
    }

def build_preprocessor(cat_cols, num_cols):
    """Build preprocessing pipeline"""
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols),
        ]
    )

def objective(trial, X_train, y_train, X_val, y_val, model_type, cat_cols, num_cols):
    """Optuna objective function for hyperparameter tuning"""
    
    # Build preprocessor
    preprocessor = build_preprocessor(cat_cols, num_cols)
    
    # Define hyperparameters based on model type
    if model_type == "random_forest":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42
        }
        model = RandomForestRegressor(**params)
        
    elif model_type == "gradient_boosting":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': 42
        }
        model = GradientBoostingRegressor(**params)
        
    else:  # decision_tree
        params = {
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
            'random_state': 42
        }
        model = DecisionTreeRegressor(**params)
    
    # Build and train pipeline
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    
    return mae

def run_enhanced_training(dry_run: bool = False):
    """
    Enhanced training pipeline with Random Forest as default model
    """
    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    required = {"scraped_at", "price", "make", "model", "year", "mileage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- Parse timestamps and choose local-day split ---
    dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["scraped_at_dt_utc"] = dt
    try:
        df["scraped_at_local"] = df["scraped_at_dt_utc"].dt.tz_convert(TIMEZONE)
    except Exception:
        df["scraped_at_local"] = df["scraped_at_dt_utc"]
    df["date_local"] = df["scraped_at_local"].dt.date

    # --- Clean numerics ---
    orig_rows = len(df)
    df["price_num"] = _clean_numeric(df["price"])
    df["year_num"] = _clean_numeric(df["year"])
    df["mileage_num"] = _clean_numeric(df["mileage"])

    valid_price_rows = int(df["price_num"].notna().sum())
    logging.info("Rows total=%d | with valid numeric price=%d", orig_rows, valid_price_rows)

    unique_dates = sorted(d for d in df["date_local"].dropna().unique())
    if len(unique_dates) < 3:
        return {"status": "noop", "reason": "need at least 3 distinct dates for train/val/test", 
                "dates": [str(d) for d in unique_dates]}

    # Split: all data before today = train+val, today = test
    today_local = unique_dates[-1]
    train_val_df = df[df["date_local"] < today_local].copy()
    test_df = df[df["date_local"] == today_local].copy()

    # Further split train_val into train (80%) and validation (20%)
    train_val_df = train_val_df[train_val_df["price_num"].notna()]
    n_trainval = len(train_val_df)
    
    if n_trainval < 100:
        return {"status": "noop", "reason": "too few training rows", "train_rows": n_trainval}
    
    # Shuffle and split
    train_val_df = train_val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.8 * n_trainval)
    train_df = train_val_df.iloc[:split_idx]
    val_df = train_val_df.iloc[split_idx:]

    logging.info("Train rows: %d, Validation rows: %d, Test rows (today=%s): %d", 
                 len(train_df), len(val_df), today_local, len(test_df))

    # --- ENHANCED FEATURE SET (10 FEATURES) ---
    target = "price_num"
    cat_cols = ["make", "model", "transmission", "color", "city", "state"]  # 6 categorical
    num_cols = ["year_num", "mileage_num"]  # 2 numeric
    feats = cat_cols + num_cols

    # Prepare data
    X_train = train_df[feats]
    y_train = train_df[target]
    X_val = val_df[feats]
    y_val = val_df[target]

    # --- HYPERPARAMETER TUNING WITH OPTUNA ---
    logging.info(f"Starting Optuna hyperparameter tuning with {N_TRIALS} trials for {MODEL_TYPE}...")
    logging.info(f"Model type: {MODEL_TYPE.upper()} (Random Forest is DEFAULT)")
    
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, MODEL_TYPE, cat_cols, num_cols),
        n_trials=N_TRIALS,
        show_progress_bar=False
    )
    
    best_params = study.best_params
    logging.info(f"Best hyperparameters: {json.dumps(best_params, indent=2)}")
    logging.info(f"Best validation MAE: ${study.best_value:.2f}")

    # --- TRAIN FINAL MODEL WITH BEST PARAMS ---
    preprocessor = build_preprocessor(cat_cols, num_cols)
    
    if MODEL_TYPE == "random_forest":
        model = RandomForestRegressor(**best_params, random_state=42)
        logging.info("✅ Training RANDOM FOREST model (default)")
    elif MODEL_TYPE == "gradient_boosting":
        model = GradientBoostingRegressor(**best_params, random_state=42)
        logging.info("✅ Training GRADIENT BOOSTING model")
    else:
        model = DecisionTreeRegressor(**best_params, random_state=42)
        logging.info("✅ Training DECISION TREE model")
    
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    # --- EVALUATE ON TEST SET (TODAY'S DATA) ---
    test_metrics = {}
    preds_df = pd.DataFrame()
    
    if not test_df.empty:
        X_test = test_df[feats]
        y_hat = pipe.predict(X_test)

        cols = ["post_id", "scraped_at", "make", "model", "year", "mileage", "price", 
                "transmission", "color", "city", "state", "zip_code"]
        available_cols = [c for c in cols if c in test_df.columns]
        preds_df = test_df[available_cols].copy()
        preds_df["actual_price"] = test_df["price_num"]
        preds_df["pred_price"] = np.round(y_hat, 2)

        test_metrics = calculate_metrics(test_df["price_num"], y_hat)

    # --- INTERPRETABILITY: PERMUTATION IMPORTANCE ---
    logging.info("Calculating permutation importance...")
    perm_result = permutation_importance(
        pipe, X_val, y_val, 
        n_repeats=10, 
        random_state=42,
        n_jobs=-1
    )
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feats,
        'importance_mean': perm_result.importances_mean,
        'importance_std': perm_result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    logging.info("Top 5 features by importance:\n%s", importance_df.head().to_string())

    # --- INTERPRETABILITY: PARTIAL DEPENDENCE PLOTS (TOP 3) ---
    logging.info("Generating Partial Dependence Plots for top 3 features...")
    top_3_features = importance_df.head(3)['feature'].tolist()
    
    # Get feature indices after preprocessing
    feature_names = (
        num_cols +  # Numeric features keep their names
        pipe.named_steps['preprocessor']
            .named_transformers_['cat']
            .named_steps['oh']
            .get_feature_names_out(cat_cols).tolist()
    )
    
    # Find indices of top features
    top_feature_indices = []
    for feat in top_3_features:
        if feat in num_cols:
            top_feature_indices.append(num_cols.index(feat))
        else:
            # For categorical, find first occurrence in encoded features
            matches = [i for i, fn in enumerate(feature_names) if fn.startswith(f"cat__{feat}_")]
            if matches:
                top_feature_indices.append(matches[0])
    
    # Generate PDP plot
    fig, axes = plt.subplots(1, min(3, len(top_feature_indices)), figsize=(15, 4))
    if len(top_feature_indices) == 1:
        axes = [axes]
    
    X_val_transformed = pipe.named_steps['preprocessor'].transform(X_val)
    
    for idx, (feat_idx, feat_name) in enumerate(zip(top_feature_indices[:3], top_3_features[:3])):
        PartialDependenceDisplay.from_estimator(
            pipe.named_steps['model'],
            X_val_transformed,
            [feat_idx],
            feature_names=[feat_name],
            ax=axes[idx] if len(top_feature_indices) > 1 else axes[0]
        )
        axes[idx].set_title(f'PDP: {feat_name}')
    
    plt.tight_layout()
    
    # Save plot to bytes
    pdp_buffer = io.BytesIO()
    plt.savefig(pdp_buffer, format='png', dpi=100, bbox_inches='tight')
    pdp_buffer.seek(0)
    plt.close()

    # --- OUTPUT PATHS (HOURLY) ---
    now_utc = pd.Timestamp.utcnow().tz_convert("UTC")
    run_timestamp = now_utc.strftime('%Y%m%d%H')
    
    preds_key = f"{OUTPUT_PREFIX}/{run_timestamp}-preds.csv"
    importance_key = f"{OUTPUT_PREFIX}/{run_timestamp}-importance.csv"
    pdp_key = f"{OUTPUT_PREFIX}/{run_timestamp}-pdp.png"
    metadata_key = f"{OUTPUT_PREFIX}/{run_timestamp}-metadata.json"

    if not dry_run:
        # Write predictions
        if len(preds_df) > 0:
            _write_csv_to_gcs(client, GCS_BUCKET, preds_key, preds_df)
            logging.info("Wrote predictions to gs://%s/%s (%d rows)", GCS_BUCKET, preds_key, len(preds_df))
        
        # Write importance
        _write_csv_to_gcs(client, GCS_BUCKET, importance_key, importance_df)
        logging.info("Wrote importance to gs://%s/%s", GCS_BUCKET, importance_key)
        
        # Write PDP plot
        _write_bytes_to_gcs(client, GCS_BUCKET, pdp_key, pdp_buffer.getvalue(), "image/png")
        logging.info("Wrote PDP plot to gs://%s/%s", GCS_BUCKET, pdp_key)
        
        # Write metadata
        metadata = {
            "timestamp": now_utc.isoformat(),
            "model_type": MODEL_TYPE,
            "best_params": best_params,
            "n_trials": N_TRIALS,
            "features_used": feats,
            "n_features": len(feats),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "test_metrics": test_metrics,
            "top_3_features": top_3_features
        }
        _write_bytes_to_gcs(
            client, GCS_BUCKET, metadata_key, 
            json.dumps(metadata, indent=2).encode(), 
            "application/json"
        )
        logging.info("Wrote metadata to gs://%s/%s", GCS_BUCKET, metadata_key)

    return {
        "status": "ok",
        "version": "enhanced-training-v1",
        "today_local": str(today_local),
        "model_type": MODEL_TYPE,
        "best_params": best_params,
        "n_trials": N_TRIALS,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "test_metrics": test_metrics,
        "features_used": feats,
        "n_features": len(feats),
        "top_3_features": top_3_features,
        "outputs": {
            "predictions": preds_key,
            "importance": importance_key,
            "pdp": pdp_key,
            "metadata": metadata_key
        },
        "dry_run": dry_run,
        "timezone": TIMEZONE,
    }

def train_enhanced_http(request):
    """HTTP entry point for enhanced training"""
    try:
        body = request.get_json(silent=True) or {}
        result = run_enhanced_training(
            dry_run=bool(body.get("dry_run", False))
        )
        code = 200 if result.get("status") == "ok" else 204
        return (json.dumps(result, indent=2), code, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s", e)
        logging.error("Trace:\n%s", traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
