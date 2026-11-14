"""
churn_utils.py
Shared constants and data-loading code used by BOTH train_model.py and app.py,
so training and prediction always use the same file paths and feature lists.
The feature lists below are copied from the tutorial code.
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "telecomechurn.csv"
MODEL_DIR = BASE_DIR / "models"
LR_MODEL_PATH = MODEL_DIR / "logistic_model.pkl"
RF_MODEL_PATH = MODEL_DIR / "rf_model.pkl"      # tuned Random Forest = the tutorial's "final model"
XGB_MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
SHAP_SUMMARY_PATH = MODEL_DIR / "shap_summary.png"
SHAP_BAR_PATH = MODEL_DIR / "shap_bar.png"

# ---- feature selection (from the tutorial) ----
TARGET = "Churn"
NUMERIC_FEATURES = ["MonthlyCharges", "TotalCharges", "tenure"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV and check that the expected columns exist."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    data = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return data


def clean_data(data: pd.DataFrame):
    """
    Tutorial cleaning: TotalCharges is text in the CSV, so convert it to numbers
    (blanks become NaN) and drop those rows.
    Returns (cleaned_data, number_of_dropped_rows).
    """
    data = data.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    original_rows = len(data)
    data = data.dropna(subset=["TotalCharges"])
    return data, original_rows - len(data)
