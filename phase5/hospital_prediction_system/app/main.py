
from fastapi import FastAPI
import joblib
import pandas as pd
from datetime import datetime
import os

from schemas import RiskRequest, ClaimRequest
from utils import prepare_features

app = FastAPI(title="Hospital Risk & Claim Intelligence API")

# Load models
risk_model = joblib.load("../models/risk_model.pkl")
claim_model = joblib.load("../models/claim_model.pkl")


@app.get("/health")
def health():
    return {"status": "API running"}


# -----------------------------
# Risk Prediction Endpoint
# -----------------------------
@app.post("/predict_risk")
def predict_risk(request: RiskRequest):

    data = request.dict()

    X = prepare_features(data, risk_model)

    prediction = risk_model.predict(X)[0]

    log_risk_prediction(data, prediction)

    return {"prediction": prediction}


# -----------------------------
# Claim Prediction Endpoint
# -----------------------------
@app.post("/predict_claim")
def predict_claim(request: ClaimRequest):

    data = request.dict()

    X = prepare_features(data, claim_model)

    prediction = claim_model.predict(X)[0]

    log_claim_prediction(data, prediction)

    return {"prediction": prediction}


# -----------------------------
# Risk Log Function
# -----------------------------
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


# -----------------------------
# Claim Log Function
# -----------------------------
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
