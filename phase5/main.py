
from fastapi import FastAPI
import joblib
import pandas as pd
from datetime import datetime
import hashlib
from schemas import RiskRequest, ClaimRequest
from utils import prepare_features

app = FastAPI(title="Hospital Risk & Claim Intelligence API")

risk_model = joblib.load("F:/AI ML/capstone/phase3/risk_model.pkl")
claim_model = joblib.load("F:/AI ML/capstone/phase3/claim_model.pkl")

@app.get("/health")
def health():
    return {"status": "API running"}

@app.post("/predict_risk")
def predict_risk(request: RiskRequest):
    data = request.dict()
    X = prepare_features(data, risk_model)
    prediction = risk_model.predict(X)[0]
    log_prediction("risk_model", data)
    return {"prediction": prediction}

@app.post("/predict_claim")
def predict_claim(request: ClaimRequest):
    data = request.dict()
    X = prepare_features(data, claim_model)
    prediction = claim_model.predict(X)[0]
    log_prediction("claim_model", data)
    return {"prediction": prediction}

def log_prediction(model_name, data):
    timestamp = datetime.now()
    feature_hash = hashlib.md5(str(data).encode()).hexdigest()
    log = {
        "timestamp": timestamp,
        "model": model_name,
        "feature_hash": feature_hash
    }
    df = pd.DataFrame([log])
    try:
        df.to_csv("prediction_logs.csv", mode="a", header=False, index=False)
    except:
        df.to_csv("prediction_logs.csv", index=False)
