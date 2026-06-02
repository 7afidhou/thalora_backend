import pickle
import joblib
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# =====================================================
# GROWTH MODELS
# =====================================================

rf_len = None
rf_w = None
xgb_len = None
xgb_w = None
lgb_len = None
lgb_w = None
growth_scaler = None

# =====================================================
# LSTM MODELS
# =====================================================

temperature_model = None
turbidity_model = None
do_model = None
ph_model = None

temperature_scaler = None
turbidity_scaler = None
do_scaler = None
ph_scaler = None


def load_growth_models():
    global rf_len, rf_w
    global xgb_len, xgb_w
    global lgb_len, lgb_w
    global growth_scaler

    with open(MODELS_DIR / "RandomForest_length.pkl", "rb") as f:
        rf_len = pickle.load(f)

    with open(MODELS_DIR / "RandomForest_weight.pkl", "rb") as f:
        rf_w = pickle.load(f)

    with open(MODELS_DIR / "XGBoost_length.pkl", "rb") as f:
        xgb_len = pickle.load(f)

    with open(MODELS_DIR / "XGBoost_weight.pkl", "rb") as f:
        xgb_w = pickle.load(f)

    with open(MODELS_DIR / "LightGBM_length.pkl", "rb") as f:
        lgb_len = pickle.load(f)

    with open(MODELS_DIR / "LightGBM_weight.pkl", "rb") as f:
        lgb_w = pickle.load(f)

    with open(MODELS_DIR / "scaler1.pkl", "rb") as f:
        growth_scaler = pickle.load(f)


def load_lstm_models():
    global temperature_model
    global turbidity_model
    global do_model
    global ph_model

    global temperature_scaler
    global turbidity_scaler
    global do_scaler
    global ph_scaler

    lstm_dir = MODELS_DIR / "lstm"

    temperature_model = load_model(
        lstm_dir / "temperature_bilstm_iotpond07.keras"
    )

    turbidity_model = load_model(
        lstm_dir / "turbidity_bilstm_iotpond07.keras"
    )

    do_model = load_model(
        lstm_dir / "do_bilstm_iotpond07.keras"
    )

    ph_model = load_model(
        lstm_dir / "ph_bilstm_iotpond07.keras"
    )

    temperature_scaler = joblib.load(
        lstm_dir / "temperature_scaler_iotpond07.pkl"
    )

    turbidity_scaler = joblib.load(
        lstm_dir / "turbidity_scaler_iotpond07.pkl"
    )

    do_scaler = joblib.load(
        lstm_dir / "do_scaler_iotpond07.pkl"
    )

    ph_scaler = joblib.load(
        lstm_dir / "ph_scaler_iotpond07.pkl"
    )


def predict_feature(model, scaler, sequence, target_idx):

    scaled = scaler.transform(sequence)

    X = np.expand_dims(scaled, axis=0)

    prediction = model.predict(
        X,
        verbose=0
    )

    dummy = np.zeros((1, 4))
    dummy[:, target_idx] = prediction.flatten()

    value = scaler.inverse_transform(dummy)[0, target_idx]

    return float(value)


def predict_all(sequence):

    return {
        "temperature": round(
            predict_feature(
                temperature_model,
                temperature_scaler,
                sequence,
                0
            ),
            3
        ),
        "turbidity": round(
            predict_feature(
                turbidity_model,
                turbidity_scaler,
                sequence,
                1
            ),
            3
        ),
        "dissolved_oxygen": round(
            predict_feature(
                do_model,
                do_scaler,
                sequence,
                2
            ),
            3
        ),
        "ph": round(
            predict_feature(
                ph_model,
                ph_scaler,
                sequence,
                3
            ),
            3
        )
    }


def predict_growth(data):

    NH3_LOW  = 0.458
    NH3_MED  = 0.616
    NH3_HIGH = 16.50

    NO3_LOW  = 150
    NO3_MED  = 383
    NO3_HIGH = 843

    default_inputs = {
        "Temperature (C)": 0.0,
        "Turbidity(NTU)": 0.0,
        "Dissolved Oxygen(g/ml)": 0.0,
        "PH": 7.0,
        "Population": 0.0
    }

    def get_numeric_value(field_name):
        raw_value = data.get(field_name, default_inputs[field_name])

        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return default_inputs[field_name]

    temp = get_numeric_value("Temperature (C)")
    turb = get_numeric_value("Turbidity(NTU)")
    do   = get_numeric_value("Dissolved Oxygen(g/ml)")
    ph   = get_numeric_value("PH")
    pop  = get_numeric_value("Population")

    # ======================================================
    # NH3 / NO3 ESTIMATION
    # ======================================================

    if turb > 15 or do < 4 or pop > 1000:
        nh3 = NH3_HIGH
        no3 = NO3_HIGH

    elif turb > 7 or do < 6:
        nh3 = NH3_MED
        no3 = NO3_MED

    else:
        nh3 = NH3_LOW
        no3 = NO3_LOW

    # ======================================================
    # TRANSFORMS
    # ======================================================

    pop  = np.log1p(max(pop, 0))
    nh3  = np.log1p(max(nh3, 0))
    no3  = np.log1p(max(no3, 0))
    turb = np.log1p(max(turb, 0))

    # ======================================================
    # FEATURE ENGINEERING
    # ======================================================

    temp_do    = temp * do
    ammonia_do = nh3 / (do + 1e-6)
    nitrate_ph = no3 * ph
    nh3_no3    = nh3 + no3
    temp_sq    = temp ** 2
    do_ph      = do * ph

    features = np.array([
        temp,
        turb,
        do,
        ph,
        nh3,
        no3,
        pop,
        temp_do,
        ammonia_do,
        nitrate_ph,
        nh3_no3,
        temp_sq,
        do_ph
    ]).reshape(1, -1)

    features_scaled = growth_scaler.transform(features)

    def predict_model(model_len, model_w):
        return {
            "length": float(
                np.expm1(model_len.predict(features_scaled))[0]
            ),
            "weight": float(
                np.expm1(model_w.predict(features_scaled))[0]
            )
        }

    rf_pred = predict_model(rf_len, rf_w)
    xgb_pred = predict_model(xgb_len, xgb_w)
    lgb_pred = predict_model(lgb_len, lgb_w)

    ensemble_length = (
        rf_pred["length"] +
        xgb_pred["length"] +
        lgb_pred["length"]
    ) / 3

    ensemble_weight = (
        rf_pred["weight"] +
        xgb_pred["weight"] +
        lgb_pred["weight"]
    ) / 3

    return {
        "Ensemble": {
            "length": ensemble_length,
            "weight": ensemble_weight
        },
        "models": {
            "RandomForest": rf_pred,
            "XGBoost": xgb_pred,
            "LightGBM": lgb_pred
        },
        "estimated_water_quality": {
            "ammonia_used": float(np.expm1(nh3)),
            "nitrate_used": float(np.expm1(no3))
        }
    }