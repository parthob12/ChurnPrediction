import pandas as pd
import numpy as np


SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")

    tenure = out["tenure"].fillna(0).astype(float)
    total = out["TotalCharges"].fillna(0).astype(float)
    monthly = out["MonthlyCharges"].fillna(0).astype(float)

    safe_tenure = tenure.replace(0, np.nan)
    out["total_charges_per_tenure"] = (total / safe_tenure).fillna(0)

    expected_total = monthly * tenure
    out["billing_drift"] = np.subtract(total, expected_total)

    service_yes = 0
    for col in SERVICE_COLS:
        if col in out.columns:
            service_yes = service_yes + (out[col].astype(str).str.lower() == "yes").astype(int)
    out["services_yes_count"] = service_yes
    out["has_multiple_services"] = (out["services_yes_count"] >= 3).astype(int)

    out["churn_label"] = (out["Churn"].astype(str).str.lower() == "yes").astype(int)

    return out


def split_xy(df: pd.DataFrame):
    y = df["churn_label"].astype(int)
    X = df.drop(columns=["Churn", "churn_label"], errors="ignore")
    return X, y
