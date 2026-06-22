import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

from src.preprocessing import TENURE_GROUP_MAP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "telco_churn_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

THRESHOLD = 0.35
RANDOM_STATE = 42

BEST_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 4,
    "num_leaves": 15,
    "min_child_samples": 30
}

def load_features() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if df["tenure_group"].dtype == object:
        df["tenure_group"] = df["tenure_group"].map(TENURE_GROUP_MAP)
    return df


def train():
    df = load_features()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    model = lgb.LGBMClassifier(
        **BEST_PARAMS, random_state=RANDOM_STATE, verbose=-1
    )
    model.fit(
        X_train_sm, y_train_sm,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Best iteration: {model.best_iteration_}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "lgbm_churn_model.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "feature_names.pkl"))

    test_data = X_test.copy()
    test_data["Churn"] = y_test
    test_data.to_csv(
        os.path.join(BASE_DIR, "data", "processed", "test_set.csv"), index=False
    )

    print("Model, feature names, and test set saved.")
    return model, roc_auc, f1


if __name__ == "__main__":
    train()