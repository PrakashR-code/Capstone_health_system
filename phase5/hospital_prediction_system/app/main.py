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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hospital Risk & Claim Intelligence API")

# Load models
risk_model = joblib.load("../models/risk_model.pkl")
claim_model = joblib.load("../models/claim_model.pkl")


@app.get("/health")
def health():
    logger.info("Health check endpoint called")
    return {"status": "API running"}



def _derive_age_group(age: int) -> str:
    if age <= 18:
        return "Child"
    elif age <= 40:
        return "Adult"
    elif age <= 60:
        return "Middle"
    return "Senior"


@app.post("/predict_risk")
def predict_risk(request: RiskRequest):
    data = request.dict()
    data["age_group"] = _derive_age_group(data["age"])
    logger.info(f"Risk request received: {data}")
    # Ensure all required fields are present (fill missing with default)
    required_fields = [
        "age", "gender", "department", "visit_type", "chronic_flag",
        "length_of_stay_hours", "visit_frequency", "avg_los_per_patient",
        "days_since_registration", "age_group", "billed_amount"
    ]
    for field in required_fields:
        if field not in data:
            data[field] = 0 if field not in ["gender", "department", "visit_type", "age_group"] else ""
    X = prepare_features(data, risk_model, "risk")
    if hasattr(risk_model, "feature_names_in_"):
        X = X[risk_model.feature_names_in_]
    prediction = risk_model.predict(X)[0]
    logger.info(f"Risk prediction: {prediction}")
    log_risk_prediction(data, prediction)
    return {"prediction": prediction}



@app.post("/predict_claim")
def predict_claim(request: ClaimRequest):
    data = request.dict()
    logger.info(f"Claim request received: {data}")
    # Ensure all required fields are present (fill missing with default)
    required_fields = [
        "age", "gender", "department", "visit_type",
        "length_of_stay_hours", "visit_frequency", "chronic_flag",
        "provider_rejection_rate", "visit_intensity", "billed_amount"
    ]
    for field in required_fields:
        if field not in data:
            data[field] = 0 if field not in ["gender", "department", "visit_type"] else ""
    X = prepare_features(data, claim_model, "claim")
    if hasattr(claim_model, "feature_names_in_"):
        X = X[claim_model.feature_names_in_]
    prediction = claim_model.predict(X)[0]
    logger.info(f"Claim prediction: {prediction}")
    log_claim_prediction(data, prediction)
    return {"prediction": prediction}


# ---------------- Risk Log Function ----------------
def log_risk_prediction(data, prediction):

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
        header=not os.path.exists(log_file),
        index=False
    )

    logger.info("Risk prediction logged to CSV")


# ---------------- Claim Log Function ----------------
def log_claim_prediction(data, prediction):

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
        header=not os.path.exists(log_file),
        index=False
    )

    logger.info("Claim prediction logged to CSV")
