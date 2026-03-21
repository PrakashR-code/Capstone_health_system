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
REFERENCE_PATH = "../data/model_table.csv"
CURRENT_PATH = "../data/new_model_table.csv"

THRESHOLD = 0.1

OUTPUT_CSV = "../data/advanced_drift_report.csv"
OUTPUT_JSON = "../data/advanced_drift_report.json"


# ---------------- DRIFT FUNCTION ----------------
def check_drift(reference_path, current_path):

    ref_df = pd.read_csv(reference_path)
    curr_df = pd.read_csv(current_path)

    drift_report = {}

    # ---------- NUMERIC DRIFT ----------
    num_cols = ref_df.select_dtypes(include=["int64", "float64"]).columns

    for col in num_cols:

        if col not in curr_df.columns:
            continue

        ref_mean = ref_df[col].mean()
        curr_mean = curr_df[col].mean()

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
    cat_cols = ref_df.select_dtypes(include=["object"]).columns

    for col in cat_cols:

        if col not in curr_df.columns:
            continue

        ref_cats = set(ref_df[col].dropna().unique())
        curr_cats = set(curr_df[col].dropna().unique())

        new_cats = curr_cats - ref_cats

        status = f"DRIFT: New categories {new_cats}" if new_cats else "Stable"

        drift_report[col] = {
            "type": "categorical",
            "status": status,
            "new_categories": list(new_cats)
        }

    return drift_report


# ---------------- SAVE REPORT ----------------
def save_report(drift_report):

    rows = []

    for col, details in drift_report.items():

        row = {"feature": col}
        row.update(details)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

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
