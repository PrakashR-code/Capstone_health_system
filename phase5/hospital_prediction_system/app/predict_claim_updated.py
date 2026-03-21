import joblib
import pandas as pd

# Load model
model = joblib.load("../models/claim_model.pkl")



# ✅ CORRECT SAMPLE INPUT (ONLY VALID FEATURES)
test = {
    "age": 45,
    "gender": "F",
    "department": "General",
    "visit_type": "OPD",
    "length_of_stay_hours": 12.5,
    "visit_frequency": 2,
    "chronic_flag": 1,
    "provider_rejection_rate": 0.15,
    "visit_intensity": 3.0,
    "billed_amount": 15000.0
}

# Convert to DataFrame
df = pd.DataFrame([test])

# Encode categoricals: claim model is a raw RandomForest (no built-in pipeline)
# Mappings recovered from training data (model_table.csv) via pd.factorize()
GENDER_MAP  = {"M": 1, "F": 0}
DEPT_MAP    = {"Cardiology": 0, "Orthopedics": 1, "ICU": 2, "General": 3, "ER": 4, "Neurology": 5}
VTYPE_MAP   = {"ER": 0, "OPD": 1, "ICU": 2}

df["gender"]     = df["gender"].map(GENDER_MAP).fillna(0).astype(int)
df["department"] = df["department"].map(DEPT_MAP).fillna(0).astype(int)
df["visit_type"] = df["visit_type"].map(VTYPE_MAP).fillna(0).astype(int)
claim_features = [
    "age", "gender", "department", "visit_type",
    "length_of_stay_hours", "visit_frequency", "chronic_flag",
    "provider_rejection_rate", "visit_intensity", "billed_amount"
]
df = df[claim_features]

print("Features:", df.columns.tolist())
# Predictions
print("Prediction:", model.predict(df))
print("Probabilities:", model.predict_proba(df))