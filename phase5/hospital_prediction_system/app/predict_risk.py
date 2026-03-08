import joblib
import pandas as pd

model = joblib.load("F:/AI ML/capstone/phase3/risk_model.pkl")

test = {
 "age":75,
 "chronic_flag":1,
 "length_of_stay_hours":48,
 "visit_frequency":8,
 "avg_los_per_patient":20,
 "provider_rejection_rate":0.6,
 "days_since_registration":100,
 "visit_month":12,
 "visit_dayofweek":6,
 "department":"ICU",
 "visit_type":"ICU"
}

df = pd.DataFrame([test])
df = pd.get_dummies(df, columns=["department","visit_type"])
df = df.reindex(columns=model.feature_names_in_, fill_value=0)

print(model.predict(df))
print(model.predict_proba(df))