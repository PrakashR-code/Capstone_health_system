import joblib
import pandas as pd

# Load model
model = joblib.load("../models/risk_model.pkl")



def derive_age_group(age):
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
test["age_group"] = derive_age_group(test["age"])


# Optionally suppress sklearn version warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Convert to DataFrame
df = pd.DataFrame([test])

# Do NOT encode categorical variables; let the pipeline handle them as strings




# Only use features from risk_features.json (order matches JSON)
risk_features = [
    "age", "gender", "department", "visit_type", "chronic_flag",
    "length_of_stay_hours", "visit_frequency", "avg_los_per_patient",
    "days_since_registration", "age_group", "billed_amount"
]  # age_group is auto-derived from age, not a user input
df = df[risk_features]

# Predictions
print("Prediction:", model.predict(df))
print("Probabilities:", model.predict_proba(df))