"""
utils.py — Feature Preparation Utilities

Handles loading the feature schema from JSON and preparing the feature
DataFrame that is passed to each model for inference.

Key Design Note
---------------
The RISK model is a full sklearn Pipeline (ColumnTransformer + OrdinalEncoder)
so it accepts raw string values for categorical columns without any pre-processing.

The CLAIM model is a raw RandomForestClassifier with NO built-in preprocessing.
Categoricals must be manually encoded before calling .predict().
The encoding maps below were reverse-engineered from the training data using
pd.factorize() on model_table.csv to match the exact integer codes seen during training.
"""

import pandas as pd
import json

# Encoding maps for the CLAIM model only (raw RandomForest, no pipeline).
# Integer codes match pd.factorize() order applied to model_table.csv during training.
_CLAIM_GENDER_MAP = {"M": 1, "F": 0}
_CLAIM_DEPT_MAP   = {"Cardiology": 0, "Orthopedics": 1, "ICU": 2, "General": 3, "ER": 4, "Neurology": 5}
_CLAIM_VTYPE_MAP  = {"ER": 0, "OPD": 1, "ICU": 2}

def load_features(model_type):
    """Read the feature list from the appropriate JSON schema file.

    Returns feature keys in the exact order they appear in the JSON,
    which must match the column order seen during model training.
    """
    if model_type == "risk":
        with open("../data/risk_features.json") as f:
            config = json.load(f)
    else:
        with open("../data/claim_features.json") as f:
            config = json.load(f)
    # Always return features in the order defined in the JSON
    return list(config["features"].keys())


def prepare_features(data, model, model_type):
    """Build the feature DataFrame ready for model.predict().

    Steps:
      1. Convert the request dict to a single-row DataFrame.
      2. Load the expected feature list from the JSON schema.
      3. Fill any missing columns with safe defaults (0 / empty string).
      4. If model_type == 'claim', apply manual categorical encoding
         so the raw RandomForest receives integer-coded inputs.

    Parameters
    ----------
    data       : dict  — request payload (already a plain dict from .dict())
    model      : sklearn model object (used only to check feature_names_in_ externally)
    model_type : str   — 'risk' or 'claim'

    Returns
    -------
    pd.DataFrame — single-row DataFrame with correct columns and dtypes
    """
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
    # Keep only required columns in training order
    df = df[feature_list]
    # Encode categoricals for claim model (no built-in pipeline)
    if model_type == "claim":
        df["gender"]     = df["gender"].map(_CLAIM_GENDER_MAP).fillna(0).astype(int)
        df["department"] = df["department"].map(_CLAIM_DEPT_MAP).fillna(0).astype(int)
        df["visit_type"] = df["visit_type"].map(_CLAIM_VTYPE_MAP).fillna(0).astype(int)
    return df