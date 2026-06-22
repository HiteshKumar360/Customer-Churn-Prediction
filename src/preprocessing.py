import pandas as pd

BINARY_MAP = {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}
CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}
INTERNET_MAP = {"No": 0, "DSL": 1, "Fiber optic": 2}
PAYMENT_MAP = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3,
}
GENDER_MAP = {"Male": 0, "Female": 1}
TENURE_GROUP_MAP = {"New": 0, "Developing": 1, "Mature": 2, "Loyal": 3}

SERVICE_COLUMNS = [
    "PhoneService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df = df.drop(columns=["customerID"], errors="ignore")
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in SERVICE_COLUMNS + ["Partner", "Dependents", "PaperlessBilling", "MultipleLines"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map(BINARY_MAP)

    if "Contract" in df.columns and df["Contract"].dtype == object:
        df["Contract"] = df["Contract"].map(CONTRACT_MAP)

    if "InternetService" in df.columns and df["InternetService"].dtype == object:
        df["InternetService"] = df["InternetService"].map(INTERNET_MAP)

    if "PaymentMethod" in df.columns and df["PaymentMethod"].dtype == object:
        df["PaymentMethod"] = df["PaymentMethod"].map(PAYMENT_MAP)

    if "gender" in df.columns and df["gender"].dtype == object:
        df["gender"] = df["gender"].map(GENDER_MAP)

    if "tenure_group" in df.columns and df["tenure_group"].dtype == object:
        df["tenure_group"] = df["tenure_group"].map(TENURE_GROUP_MAP)

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["New", "Developing", "Mature", "Loyal"],
    ).map(TENURE_GROUP_MAP)

    df["num_services"] = df[SERVICE_COLUMNS].sum(axis=1)
    df["is_new_customer"] = (df["tenure"] <= 12).astype(int)
    df["CLV"] = df["tenure"] * df["MonthlyCharges"]
    df["avg_service_cost"] = df["MonthlyCharges"] / (df["num_services"] + 1)
    df["high_risk_flag"] = (
        (df["Contract"] == 0) & (df["is_new_customer"] == 1)
    ).astype(int)

    return df


def build_customer_row(raw_inputs: dict) -> dict:
    tenure = raw_inputs["tenure"]
    monthly_charges = raw_inputs["monthly_charges"]
    total_charges = tenure * monthly_charges

    services_flags = {
        "PhoneService": BINARY_MAP[raw_inputs["phone_service"]],
        "OnlineSecurity": BINARY_MAP[raw_inputs["online_security"]],
        "OnlineBackup": BINARY_MAP[raw_inputs["online_backup"]],
        "DeviceProtection": BINARY_MAP[raw_inputs["device_protection"]],
        "TechSupport": BINARY_MAP[raw_inputs["tech_support"]],
        "StreamingTV": BINARY_MAP[raw_inputs["streaming_tv"]],
        "StreamingMovies": BINARY_MAP[raw_inputs["streaming_movies"]],
    }
    num_services = sum(services_flags.values())

    if tenure <= 12:
        tenure_group = 0
    elif tenure <= 24:
        tenure_group = 1
    elif tenure <= 48:
        tenure_group = 2
    else:
        tenure_group = 3

    is_new_customer = int(tenure <= 12)
    clv = tenure * monthly_charges
    avg_service_cost = monthly_charges / (num_services + 1)
    contract_code = CONTRACT_MAP[raw_inputs["contract"]]
    high_risk_flag = int((contract_code == 0) and (is_new_customer == 1))

    row = {
        "gender": GENDER_MAP[raw_inputs["gender"]],
        "SeniorCitizen": BINARY_MAP[raw_inputs["senior_citizen"]],
        "Partner": BINARY_MAP[raw_inputs["partner"]],
        "Dependents": BINARY_MAP[raw_inputs["dependents"]],
        "tenure": tenure,
        "PhoneService": services_flags["PhoneService"],
        "MultipleLines": BINARY_MAP[raw_inputs["multiple_lines"]],
        "InternetService": INTERNET_MAP[raw_inputs["internet_service"]],
        "OnlineSecurity": services_flags["OnlineSecurity"],
        "OnlineBackup": services_flags["OnlineBackup"],
        "DeviceProtection": services_flags["DeviceProtection"],
        "TechSupport": services_flags["TechSupport"],
        "StreamingTV": services_flags["StreamingTV"],
        "StreamingMovies": services_flags["StreamingMovies"],
        "Contract": contract_code,
        "PaperlessBilling": BINARY_MAP[raw_inputs["paperless"]],
        "PaymentMethod": PAYMENT_MAP[raw_inputs["payment_method"]],
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "tenure_group": tenure_group,
        "num_services": num_services,
        "is_new_customer": is_new_customer,
        "CLV": clv,
        "avg_service_cost": avg_service_cost,
        "high_risk_flag": high_risk_flag,
    }
    return row