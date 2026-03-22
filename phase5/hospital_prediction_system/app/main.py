"""
main.py — FastAPI Application Entry Point

Hospital Risk & Claim Intelligence Platform
--------------------------------------------
Exposes two prediction endpoints:
  POST /predict_risk  — Visit risk classification (Low / Medium / High)
  POST /predict_claim — Insurance claim outcome prediction (Paid / Pending / Rejected)

Models are loaded once at startup from pre-trained .pkl files.
All predictions are appended to CSV audit logs for monitoring.
"""

from fastapi import FastAPI
import joblib
from numpy import rint
import pandas as pd
from datetime import datetime
import os
import logging

from schemas import RiskRequest, ClaimRequest
from utils import prepare_features

# ---------------- Logging Setup ----------------
# Structured logs help trace prediction requests in production (AWS CloudWatch etc.)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hospital Risk & Claim Intelligence API")

# Module-level globals; populated during startup so route handlers can access them
risk_model = None
claim_model = None


@app.on_event("startup")
def load_model():
    """Load both trained models once when the server starts.

    risk_model  — sklearn Pipeline (handles string categoricals natively).
    claim_model — raw RandomForestClassifier (requires manual encoding via utils.py).
    """
    global risk_model, claim_model
    print("Loading models...")
    risk_model = joblib.load("../models/risk_model.pkl")
    claim_model = joblib.load("../models/claim_model.pkl")
    print("✅ Models loaded")

@app.get("/health")
def health():
    """Liveness probe used by load balancers and deployment health checks."""
    logger.info("Health check endpoint called")
    return {"status": "API running"}



def _derive_age_group(age: int) -> str:
    """Convert numeric age to age-group category expected by the risk model.

    age_group is NOT exposed to end users; it is derived automatically here
    so the API accepts only 10 clean features instead of 11.
    """
    if age <= 18:
        return "Child"
    elif age <= 40:
        return "Adult"
    elif age <= 60:
        return "Middle"
    return "Senior"


@app.post("/predict_risk")
def predict_risk(request: RiskRequest):
    """Predict visit risk level (Low / Medium / High) for a hospital visit.

    age_group is auto-derived from age before passing to the model,
    keeping the public API surface clean (10 inputs, no derived fields).
    Returns the predicted risk label as a JSON string.
    """
    data = request.dict()
    # Derive age_group internally — not exposed in the request schema
    data["age_group"] = _derive_age_group(data["age"])
    logger.info(f"Risk request received: {data}")
    # Guard: ensure all required fields are present (fill missing with defaults)
    required_fields = [
        "age", "gender", "department", "visit_type", "chronic_flag",
        "length_of_stay_hours", "visit_frequency", "avg_los_per_patient",
        "days_since_registration", "age_group", "billed_amount"
    ]
    for field in required_fields:
        if field not in data:
            data[field] = 0 if field not in ["gender", "department", "visit_type", "age_group"] else ""
    X = prepare_features(data, risk_model, "risk")
    # Reorder columns to match the exact order the pipeline was trained on
    if hasattr(risk_model, "feature_names_in_"):
        X = X[risk_model.feature_names_in_]
    prediction = risk_model.predict(X)[0]
    logger.info(f"Risk prediction: {prediction}")
    log_risk_prediction(data, prediction)
    return {"prediction": prediction}



@app.post("/predict_claim")
def predict_claim(request: ClaimRequest):
    """Predict insurance claim outcome (Paid / Pending / Rejected) for a hospital visit.

    The claim model is a raw RandomForestClassifier with NO built-in preprocessing.
    Categorical encoding (gender, department, visit_type) is applied in utils.prepare_features()
    using factorize-derived mappings from the original training dataset.
    """
    data = request.dict()
    logger.info(f"Claim request received: {data}")
    # Guard: ensure all required fields are present (fill missing with defaults)
    required_fields = [
        "age", "gender", "department", "visit_type",
        "length_of_stay_hours", "visit_frequency", "chronic_flag",
        "provider_rejection_rate", "visit_intensity", "billed_amount"
    ]
    for field in required_fields:
        if field not in data:
            data[field] = 0 if field not in ["gender", "department", "visit_type"] else ""
    # prepare_features handles categorical encoding for claim model
    X = prepare_features(data, claim_model, "claim")
    # Reorder columns to match the exact order the model was trained on
    if hasattr(claim_model, "feature_names_in_"):
        X = X[claim_model.feature_names_in_]
    prediction = claim_model.predict(X)[0]
    logger.info(f"Claim prediction: {prediction}")
    log_claim_prediction(data, prediction)
    return {"prediction": prediction}


# ---------------- Risk Log Function ----------------
def log_risk_prediction(data, prediction):
    """Append a single risk prediction event to the audit log CSV.

    The log is used for drift monitoring and post-deployment analysis.
    Header row is written only when the file does not yet exist.
    """
    log = {
        "timestamp": datetime.now(),
        "prediction": prediction,
        **data
    }

    df = pd.DataFrame([log])

    log_file = "../logs/risk_predictions_log.csv"

    df.to_csv(
        log_file,
        mode="a",
        header=not os.path.exists(log_file),  # write header only on first entry
        index=False
    )

    logger.info("Risk prediction logged to CSV")


# ---------------- Claim Log Function ----------------
def log_claim_prediction(data, prediction):
    """Append a single claim prediction event to the audit log CSV.

    Rejected-claim predictions are especially important to track;
    the log enables revenue-leakage analysis over time.
    """
    log = {
        "timestamp": datetime.now(),
        "prediction": prediction,
        **data
    }

    df = pd.DataFrame([log])

    log_file = "../logs/claim_predictions_log.csv"

    df.to_csv(
        log_file,
        mode="a",
        header=not os.path.exists(log_file),  # write header only on first entry
        index=False
    )

    logger.info("Claim prediction logged to CSV")
