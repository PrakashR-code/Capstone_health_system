
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

# Path to the original training dataset — used as the baseline reference for drift
TRAIN_DATA_PATH = "../data/model_table.csv"

# Paths to prediction logs written by the FastAPI app during inference
RISK_LOG_PATH = "../logs/risk_predictions_log.csv"
CLAIM_LOG_PATH = "../logs/claim_predictions_log.csv"

# A mean deviation greater than this value flags a feature as drifted.
# 0.1 = 10% absolute shift from the training baseline mean.
DRIFT_THRESHOLD = 0.1

# Output CSV summarising all drift findings for this monitoring run
REPORT_FILE = "../data/drift_report.csv"

# ------------------------------------------------
# Feature Drift Detection
# ------------------------------------------------

def detect_feature_drift(train_df, log_df):
    """Compare numeric feature distributions between training data and live prediction logs.

    Uses mean deviation as a simple, interpretable proxy for distribution shift.
    Returns a list of row-dicts that will be written to drift_report.csv.
    """
    # Keep only numeric columns from each dataset for statistical comparison
    train_numeric = train_df.select_dtypes(include=["int64","float64"])
    log_numeric = log_df.select_dtypes(include=["int64","float64"])

    # Only compare columns present in both — log may have fewer columns than training data
    common_cols = list(set(train_numeric.columns) & set(log_numeric.columns))

    rows = []

    for col in common_cols:
        # Baseline mean from the original training dataset
        train_mean = train_numeric[col].mean()

        # Mean from recent live predictions (inference log)
        new_mean = log_numeric[col].mean()

        # Absolute mean deviation — simple, interpretable drift signal
        diff = abs(train_mean - new_mean)

        # Flag as drifted if deviation exceeds the configured threshold
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
    """Track the distribution of model output labels over time.

    Unlike feature drift (which checks inputs), this monitors outputs —
    e.g. if the share of 'Rejected' predictions suddenly spikes, it may
    indicate a real-world shift even if input features look stable.
    Marked 'monitor_only' because there is no labeled ground truth to diff against.
    """
    rows = []

    # Skip if the log file doesn't contain a prediction column
    if "prediction" not in df.columns:
        return rows

    # Compute the proportion of each predicted class label
    dist = df["prediction"].value_counts(normalize=True)

    for label, value in dist.items():

        rows.append({
            "model":model_name,
            "type":"prediction_distribution",
            "feature":label,          # the predicted class (e.g. 'Paid', 'Rejected')
            "train_mean":"NA",        # no training baseline for output distribution
            "new_mean":value,         # proportion of this label in recent predictions
            "difference":"NA",
            "drift_detected":"monitor_only"  # informational only — no threshold applied
        })

    return rows


# ------------------------------------------------
# Monitoring Pipeline
# ------------------------------------------------

def main():
    """Entry point — orchestrates drift detection for both models and saves the report."""

    print("\nHospital AI Monitoring System\n")

    report_rows = []  # accumulates drift rows from all checks

    # Cannot compute drift without the training baseline
    if not os.path.exists(TRAIN_DATA_PATH):
        print("Training dataset missing")
        return

    train_df = pd.read_csv(TRAIN_DATA_PATH)

    # ---------------- Risk Model ----------------
    # Feature drift is checked for the risk model because it uses a full
    # feature pipeline; input distributions are more likely to shift over time.

    if os.path.exists(RISK_LOG_PATH):

        risk_logs = pd.read_csv(RISK_LOG_PATH)

        print("\nMonitoring Risk Model")

        # Check if any input feature means have shifted from training baseline
        report_rows += detect_feature_drift(train_df, risk_logs)

        # Track the prediction output label distribution
        report_rows += detect_prediction_drift(risk_logs,"risk_model")

    else:

        print("Risk log missing")

    # ---------------- Claim Model ----------------
    # For the claim model, only prediction distribution is monitored here.
    # Feature drift for claims is covered by advanced_monitoring.py.

    if os.path.exists(CLAIM_LOG_PATH):

        claim_logs = pd.read_csv(CLAIM_LOG_PATH)

        print("\nMonitoring Claim Model")

        # Track proportion of Paid / Pending / Rejected in recent claim predictions
        report_rows += detect_prediction_drift(claim_logs,"claim_model")

    else:

        print("Claim log missing")

    # ---------------- Save Report ----------------

    if len(report_rows) > 0:

        report_df = pd.DataFrame(report_rows)

        # Persist the full drift report for downstream review / Phase 6 reporting
        report_df.to_csv(REPORT_FILE,index=False)

        print("\nDrift report generated:",REPORT_FILE)

        print("\nSummary Table:\n")

        # Print a preview — full report is in the CSV
        print(report_df.head(10))

    else:

        print("No monitoring data available")

    print("\nMonitoring completed\n")


if __name__ == "__main__":

    main()
