import joblib
import numpy as np
import pandas as pd
from pathlib import Path
    
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

_assets = None


def load_assets():
    global _assets

    if _assets is None:
        try:
            xgb_length = joblib.load(MODELS_DIR / "xgb_length.pkl")
            xgb_weight = joblib.load(MODELS_DIR / "xgb_weight.pkl")
            scaler = joblib.load(MODELS_DIR / "scaler.pkl")
            scaler_lstm = joblib.load(MODELS_DIR / "scaler_lstm.pkl")
            feature_columns = joblib.load(MODELS_DIR / "features.pkl")
        except ModuleNotFoundError as exc:
            missing_module = exc.name or "a required package"
            raise RuntimeError(
                "Missing dependency while loading saved models. "
                f"Install `{missing_module}` with pip."
            ) from exc

        try:
            from tensorflow import keras
        except ModuleNotFoundError as exc:
            missing_module = exc.name or "tensorflow"
            raise RuntimeError(
                "LSTM models require TensorFlow/Keras. "
                f"Install `{missing_module}` with pip."
            ) from exc

        lstm_length = keras.models.load_model(MODELS_DIR / "lstm_length_model.keras")
        lstm_weight = keras.models.load_model(MODELS_DIR / "lstm_weight_model.keras")

        _assets = {
            "xgb_length": xgb_length,
            "xgb_weight": xgb_weight,
            "scaler": scaler,
            "scaler_lstm": scaler_lstm,
            "feature_columns": feature_columns,
            "lstm_length": lstm_length,
            "lstm_weight": lstm_weight,
        }

    return _assets


def prepare_features(input_dict):
    df = pd.DataFrame([input_dict])

    # === CLIPPING ===
    df["Temperature (C)"] = df["Temperature (C)"].clip(lower=20)
    df["PH"] = df["PH"].clip(5, 12)
    df["Ammonia(g/ml)"] = df["Ammonia(g/ml)"].clip(upper=10)
    df["Nitrate(g/ml)"] = df["Nitrate(g/ml)"].clip(upper=2000)

    # === LOG TRANSFORMS ===
    df["Population"] = np.log1p(df["Population"])
    df["Ammonia(g/ml)"] = np.log1p(df["Ammonia(g/ml)"])
    df["Nitrate(g/ml)"] = np.log1p(df["Nitrate(g/ml)"])
    df["Turbidity(NTU)"] = np.log1p(df["Turbidity(NTU)"])

    # === FEATURE ENGINEERING ===
    df["Temp_DO"] = df["Temperature (C)"] * df["Dissolved Oxygen(g/ml)"]
    df["Ammonia_DO"] = df["Ammonia(g/ml)"] * df["Dissolved Oxygen(g/ml)"]
    df["Nitrate_PH"] = df["Nitrate(g/ml)"] * df["PH"]
    df["NH3_NO3"] = df["Ammonia(g/ml)"] * df["Nitrate(g/ml)"]
    df["Temp_squared"] = df["Temperature (C)"] ** 2
    df["DO_PH"] = df["Dissolved Oxygen(g/ml)"] * df["PH"]

    return df


def predict(input_data):
    assets = load_assets()

    scaler = assets["scaler"]
    xgb_length = assets["xgb_length"]
    xgb_weight = assets["xgb_weight"]
    scaler_lstm = assets["scaler_lstm"]
    feature_columns = assets["feature_columns"]
    lstm_length = assets["lstm_length"]
    lstm_weight = assets["lstm_weight"]

    # ================= FAST FEATURE PREP =================
    df = prepare_features(input_data)

    base_features = [
        'Temperature (C)', 'Turbidity(NTU)', 'Dissolved Oxygen(g/ml)',
        'PH', 'Ammonia(g/ml)', 'Nitrate(g/ml)', 'Population',
        'Temp_DO', 'Ammonia_DO', 'Nitrate_PH', 'NH3_NO3',
        'Temp_squared', 'DO_PH'
    ]

    # Convert to numpy immediately (faster than pandas later)
    X_base = df[base_features].values.astype("float32")

    # ================= XGBOOST =================
    X_scaled = scaler.transform(X_base)

    log_length = xgb_length.predict(X_scaled)
    length = np.expm1(log_length)

    X_weight = np.hstack([
        X_scaled,
        log_length.reshape(-1, 1),
        length.reshape(-1, 1)
    ])

    log_weight = xgb_weight.predict(X_weight)
    weight = np.expm1(log_weight)

    xgb_length_value = float(length[0])
    xgb_weight_value = float(weight[0])

    # ================= LSTM =================
    # Convert to numpy early
    X_lstm_base = df[feature_columns].values.astype("float32")
    X_lstm_scaled = scaler_lstm.transform(X_lstm_base)

    # 🔥 FAST reshape (no dynamic checks)
    X_lstm_length = X_lstm_scaled.reshape(1, 1, -1)

    lstm_log_length = lstm_length.predict(X_lstm_length, verbose=0)[0][0]
    lstm_length_value = float(np.expm1(lstm_log_length))

    # Prepare weight input
    X_lstm_weight = np.hstack([
        X_lstm_scaled,
        np.array([[lstm_log_length]], dtype="float32"),
        np.array([[lstm_length_value]], dtype="float32")
    ]).reshape(1, 1, -1)

    lstm_log_weight = lstm_weight.predict(X_lstm_weight, verbose=0)[0][0]
    lstm_weight_value = float(np.expm1(lstm_log_weight))

    return {
        "xgboost": {"length": xgb_length_value, "weight": xgb_weight_value},
        "lstm": {"length": lstm_length_value, "weight": lstm_weight_value},
    }