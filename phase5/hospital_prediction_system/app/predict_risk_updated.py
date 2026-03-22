"""
predict_risk_updated.py — Standalone Risk Prediction Script

Purpose
-------
Quick smoke-test for the risk model outside of the FastAPI server.
Run this directly to verify the model loads and produces a valid prediction
for a sample input without starting the full API.

Model Notes
-----------
risk_model.pkl is a full sklearn Pipeline (ColumnTransformer + OrdinalEncoder + RandomForest).
Categorical columns can be passed as raw strings — the pipeline handles encoding internally.
age_group is the only derived field: it is computed from age before prediction.
"""

import joblib
import pandas as pd

# Load the trained risk model from disk
model = joblib.load("../models/risk_model.pkl")



def derive_age_group(age):
    """Map numeric age to the categorical age-group label expected by the risk model."""
    if age <= 18:
        return "Child"
    elif age <= 40:
        return "Adult"
    elif age <= 60:
        return "Middle"
    return "Senior"


# ✅ CORRECT SAMPLE INPUT (ONLY VALID FEATURES)
test = {
    "age": 60,
    "gender": "M",
    "department": "ICU",
    "visit_type": "ICU",
    "chronic_flag": 1,
    "length_of_stay_hours": 24,
    "visit_frequency": 5,
    "avg_los_per_patient": 18,
    "days_since_registration": 900,
    "billed_amount": 20000.0
}
# Derive age_group from age — not exposed as a user input
test["age_group"] = derive_age_group(test["age"])


# Optionally suppress sklearn version warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Convert to DataFrame (model expects 2D input)
df = pd.DataFrame([test])

# The risk pipeline includes OrdinalEncoder, so raw strings are fine here
# Do NOT encode categorical variables; let the pipeline handle them as strings



# Select features in the exact order defined in risk_features.json
risk_features = [
    "age", "gender", "department", "visit_type", "chronic_flag",
    "length_of_stay_hours", "visit_frequency", "avg_los_per_patient",
    "days_since_registration", "age_group", "billed_amount"
]  # age_group is auto-derived from age, not a user input
df = df[risk_features]

# Predictions
print("Prediction:", model.predict(df))           # e.g. ['Low'], ['Medium'], ['High']
print("Probabilities:", model.predict_proba(df))  # Per-class confidence scores