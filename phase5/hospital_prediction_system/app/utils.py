import pandas as pd
import json

# Encoding maps for claim model (raw RandomForest; no built-in preprocessing pipeline).
# Mappings match pd.factorize() order applied to model_table.csv during training.
_CLAIM_GENDER_MAP = {"M": 1, "F": 0}
_CLAIM_DEPT_MAP   = {"Cardiology": 0, "Orthopedics": 1, "ICU": 2, "General": 3, "ER": 4, "Neurology": 5}
_CLAIM_VTYPE_MAP  = {"ER": 0, "OPD": 1, "ICU": 2}

def load_features(model_type):
    if model_type == "risk":
        with open("../data/risk_features.json") as f:
            config = json.load(f)
    else:
        with open("../data/claim_features.json") as f:
            config = json.load(f)
    # Always return features in the order defined in the JSON
    return list(config["features"].keys())


def prepare_features(data, model, model_type):
    df = pd.DataFrame([data])
    feature_list = load_features(model_type)
    # Add missing columns with default value
    for col in feature_list:
        if col not in df.columns:
            # Use empty string for categorical, 0 for numeric
            if model_type == "risk":
                with open("../data/risk_features.json") as f:
                    config = json.load(f)
            else:
                with open("../data/claim_features.json") as f:
                    config = json.load(f)
            dtype = config["features"][col]["dtype"]
            if dtype == "category":
                df[col] = ""
            else:
                df[col] = 0
    # Keep only required columns
    df = df[feature_list]
    # Encode categoricals for claim model (no built-in pipeline)
    if model_type == "claim":
        df["gender"]     = df["gender"].map(_CLAIM_GENDER_MAP).fillna(0).astype(int)
        df["department"] = df["department"].map(_CLAIM_DEPT_MAP).fillna(0).astype(int)
        df["visit_type"] = df["visit_type"].map(_CLAIM_VTYPE_MAP).fillna(0).astype(int)
    return df