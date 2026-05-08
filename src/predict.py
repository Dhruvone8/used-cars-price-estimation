import numpy as np
import pandas as pd
from datetime import datetime
import joblib

FEATURE_COLUMNS = [
    'Brand', 'Km', 'Owner', 'Location', 'Engine', 'Seats',
    'Base Model', 'Car Age', 'Km Per Year',
    'Diesel', 'Petrol', 'Manual Transmission'
]

def get_feature_columns():
    return FEATURE_COLUMNS

def load_artifacts(model_path: str, encoders_path: str):
    ml_model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    return ml_model, encoders

def prepare_input(raw: dict, encoders: dict) -> pd.DataFrame:
    """
    raw keys:
      Brand, Model_Base, Year, Km, Fuel,
      Transmission, Owner, Location, Engine, Seats
    Returns single-row DataFrame ready for model prediction.
    """
    current_year = datetime.now().year  
    car_age      = current_year - int(raw['Year'])
    km_per_year = float(raw['Km']) / max(car_age, 1)

    # ── Label encode ────────────────────────────────
    brand_enc = encoders['Brand']
    brand_val = (
        int(brand_enc.transform([raw['Brand']])[0])
        if raw['Brand'] in brand_enc.classes_ else -1
    )

    model_enc = encoders['Model_Base']
    model_val = (
        int(model_enc.transform([raw['Model_Base']])[0])
        if raw['Model_Base'] in model_enc.classes_ else -1
    )

    loc_enc = encoders['Location']
    loc_val = (
        int(loc_enc.transform([raw['Location']])[0])
        if raw['Location'] in loc_enc.classes_ else -1
    )

    # ── One-hot ──────────────────────────────────────
    is_diesel = 1 if raw['Fuel'] == 'Diesel' else 0
    is_petrol = 1 if raw['Fuel'] == 'Petrol' else 0
    is_manual = 1 if raw['Transmission'] == 'Manual' else 0

    row = {
        'Brand':               brand_val,
        'Km':                  float(raw['Km']),
        'Owner':               int(raw['Owner']),
        'Location':            loc_val,
        'Engine':              float(raw['Engine']),
        'Seats':               float(raw['Seats']),
        'Base Model':          model_val,
        'Car Age':             car_age,
        'Km Per Year':         km_per_year,
        'Diesel':              is_diesel,
        'Petrol':              is_petrol,
        'Manual Transmission': is_manual,
    }

    return pd.DataFrame([row])[FEATURE_COLUMNS]


def predict_price(ml_model, input_df: pd.DataFrame) -> float:
    log_price = ml_model.predict(input_df.values.astype(float))[0]
    return float(np.expm1(log_price))


def predict_range(predicted_price: float, mape: float = 0.103) -> tuple:
    """
    Return (low, high) price range using model MAPE of 10.3%.
    Gives buyer and seller a realistic negotiation band.
    """
    low  = int(predicted_price * (1 - mape))
    high = int(predicted_price * (1 + mape))
    return low, high