"""
Advanced Monitoring Script (Batch Drift Detection)
------------------------------------------------
Compares NEW incoming model_table.csv with reference (training data)

Supports:
1. Numeric drift (mean deviation)
2. Categorical drift (new categories)
3. Generates detailed JSON + CSV report

Usage:
    python advanced_monitoring.py
"""

import pandas as pd
import json
import os

# ---------------- CONFIG ----------------

# Reference dataset — the original training data used as the stable baseline
REFERENCE_PATH = "../data/model_table.csv"

# New incoming data — a fresh batch of patient/visit records to compare against the baseline
# Replace this file with each new batch before running the script
CURRENT_PATH = "../data/new_model_table.csv"

# Drift threshold for numeric features: absolute mean deviation above this is flagged
THRESHOLD = 0.1

# Output files — CSV for human review, JSON for programmatic consumption
OUTPUT_CSV = "../data/advanced_drift_report.csv"
OUTPUT_JSON = "../data/advanced_drift_report.json"


# ---------------- DRIFT FUNCTION ----------------
def check_drift(reference_path, current_path):
    """Compare a new data batch against the training reference for both numeric and categorical drift.

    Numeric drift  — flags columns where the absolute mean has shifted by more than THRESHOLD.
    Categorical drift — flags columns where the new batch contains category values
                        not seen during training (e.g. a new department or insurance provider).
    Returns a dict keyed by column name containing drift details.
    """
    ref_df  = pd.read_csv(reference_path)
    curr_df = pd.read_csv(current_path)

    drift_report = {}  # keyed by column name

    # ---------- NUMERIC DRIFT ----------
    # Select all integer and float columns from the reference dataset
    num_cols = ref_df.select_dtypes(include=["int64", "float64"]).columns

    for col in num_cols:

        # Skip columns absent from the new batch — can't compare
        if col not in curr_df.columns:
            continue

        ref_mean  = ref_df[col].mean()   # training baseline mean
        curr_mean = curr_df[col].mean()  # mean in the new batch

        # Absolute mean deviation as the drift signal
        deviation = abs(curr_mean - ref_mean)

        status = "DRIFT DETECTED" if deviation > THRESHOLD else "Stable"

        drift_report[col] = {
            "type": "numeric",
            "ref_mean": ref_mean,
            "curr_mean": curr_mean,
            "deviation": deviation,
            "status": status
        }

    # ---------- CATEGORICAL DRIFT ----------
    # Select all object (string) columns — department, visit_type, gender, etc.
    cat_cols = ref_df.select_dtypes(include=["object"]).columns

    for col in cat_cols:

        # Skip columns absent from the new batch
        if col not in curr_df.columns:
            continue

        ref_cats  = set(ref_df[col].dropna().unique())   # categories seen in training
        curr_cats = set(curr_df[col].dropna().unique())  # categories in new batch

        # New categories the model has never seen — could cause encoding errors at inference
        new_cats = curr_cats - ref_cats

        status = f"DRIFT: New categories {new_cats}" if new_cats else "Stable"

        drift_report[col] = {
            "type": "categorical",
            "status": status,
            "new_categories": list(new_cats)  # empty list if no drift
        }

    return drift_report


# ---------------- SAVE REPORT ----------------
def save_report(drift_report):
    """Persist the drift report in two formats:
    - CSV  : flat table, easy to open in Excel / attach to capstone report
    - JSON : nested structure, easy to parse programmatically or send to an alerting system
    """
    rows = []

    # Flatten the nested dict into rows so it can be written as a CSV
    for col, details in drift_report.items():

        row = {"feature": col}  # feature name as the first column
        row.update(details)     # merge in type, status, deviation, etc.

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)  # flat CSV for human review

    # JSON preserves the original nested structure for programmatic use
    with open(OUTPUT_JSON, "w") as f:
        json.dump(drift_report, f, indent=4)

    print("\nDrift reports generated:")
    print("CSV:", OUTPUT_CSV)
    print("JSON:", OUTPUT_JSON)


# ---------------- MAIN ----------------
if __name__ == "__main__":

    print("\nRunning Batch Drift Detection...\n")

    if not os.path.exists(REFERENCE_PATH):
        print("Reference file missing")
        exit()

    if not os.path.exists(CURRENT_PATH):
        print("New dataset missing")
        exit()

    report = check_drift(REFERENCE_PATH, CURRENT_PATH)

    save_report(report)

    print("\nDrift Detection Completed\n")
