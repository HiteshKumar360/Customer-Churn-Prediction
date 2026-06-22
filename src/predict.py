import os
import joblib
import pandas as pd

from src.preprocessing import build_customer_row

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lgbm_churn_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")

# Best threshold selected in notebooks/03_model.ipynb (highest F1, 79% recall)
DEFAULT_THRESHOLD = 0.35


def load_artifacts():
    """Loads the trained model and the exact feature column order it expects."""
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, feature_names


def predict_single(raw_inputs: dict, model=None, feature_names=None,
                    threshold: float = DEFAULT_THRESHOLD) -> dict:
    if model is None or feature_names is None:
        model, feature_names = load_artifacts()

    row = build_customer_row(raw_inputs)
    input_df = pd.DataFrame([row])

    # Guard against any missing engineered columns (defensive, should be empty)
    missing = [f for f in feature_names if f not in input_df.columns]
    for m in missing:
        input_df[m] = 0
    input_df = input_df[feature_names]

    churn_proba = float(model.predict_proba(input_df)[0][1])
    will_churn = churn_proba >= threshold

    return {
        "churn_probability": round(churn_proba, 4),
        "will_churn": bool(will_churn),
        "threshold_used": threshold,
    }


def predict_batch(df: pd.DataFrame, model=None, feature_names=None,
                   threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    if model is None or feature_names is None:
        model, feature_names = load_artifacts()

    X = df[feature_names]
    df = df.copy()
    df["churn_probability"] = model.predict_proba(X)[:, 1]
    df["will_churn"] = (df["churn_probability"] >= threshold).astype(int)
    return df