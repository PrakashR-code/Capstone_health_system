
"""
Phase 6 — Monitoring, Drift Detection, and Governance
----------------------------------------------------
This script performs:
1. Feature drift detection (risk model)
2. Prediction drift detection (risk + claim models)
3. Generates drift_report.csv for reporting

Usage:
    python phase6_monitoring_report.py
"""

import pandas as pd
import os

# ------------------------------------------------
# Configuration
# ------------------------------------------------
TRAIN_DATA_PATH = "../data/model_table.csv"
RISK_LOG_PATH = "../logs/risk_predictions_log.csv"
CLAIM_LOG_PATH = "../logs/claim_predictions_log.csv"

DRIFT_THRESHOLD = 0.1
REPORT_FILE = "../data/drift_report.csv"

# ------------------------------------------------
# Feature Drift Detection
# ------------------------------------------------

def detect_feature_drift(train_df, log_df):

    train_numeric = train_df.select_dtypes(include=["int64","float64"])
    log_numeric = log_df.select_dtypes(include=["int64","float64"])

    common_cols = list(set(train_numeric.columns) & set(log_numeric.columns))

    rows = []

    for col in common_cols:

        train_mean = train_numeric[col].mean()
        new_mean = log_numeric[col].mean()

        diff = abs(train_mean - new_mean)

        drift = diff > DRIFT_THRESHOLD

        rows.append({
            "model":"risk_model",
            "type":"feature_drift",
            "feature":col,
            "train_mean":train_mean,
            "new_mean":new_mean,
            "difference":diff,
            "drift_detected":drift
        })

    return rows

# ------------------------------------------------
# Prediction Drift Detection
# ------------------------------------------------

def detect_prediction_drift(df, model_name):

    rows = []

    if "prediction" not in df.columns:
        return rows

    dist = df["prediction"].value_counts(normalize=True)

    for label, value in dist.items():

        rows.append({
            "model":model_name,
            "type":"prediction_distribution",
            "feature":label,
            "train_mean":"NA",
            "new_mean":value,
            "difference":"NA",
            "drift_detected":"monitor_only"
        })

    return rows


# ------------------------------------------------
# Monitoring Pipeline
# ------------------------------------------------

def main():

    print("\nHospital AI Monitoring System\n")

    report_rows = []

    if not os.path.exists(TRAIN_DATA_PATH):
        print("Training dataset missing")
        return

    train_df = pd.read_csv(TRAIN_DATA_PATH)

    # ---------------- Risk Model ----------------

    if os.path.exists(RISK_LOG_PATH):

        risk_logs = pd.read_csv(RISK_LOG_PATH)

        print("\nMonitoring Risk Model")

        report_rows += detect_feature_drift(train_df, risk_logs)

        report_rows += detect_prediction_drift(risk_logs,"risk_model")

    else:

        print("Risk log missing")

    # ---------------- Claim Model ----------------

    if os.path.exists(CLAIM_LOG_PATH):

        claim_logs = pd.read_csv(CLAIM_LOG_PATH)

        print("\nMonitoring Claim Model")

        report_rows += detect_prediction_drift(claim_logs,"claim_model")

    else:

        print("Claim log missing")

    # ---------------- Save Report ----------------

    if len(report_rows) > 0:

        report_df = pd.DataFrame(report_rows)

        report_df.to_csv(REPORT_FILE,index=False)

        print("\nDrift report generated:",REPORT_FILE)

        print("\nSummary Table:\n")

        print(report_df.head(10))

    else:

        print("No monitoring data available")

    print("\nMonitoring completed\n")


if __name__ == "__main__":

    main()
