"""
build_features.py
-----------------
Feature engineering script for Phase 2 of the Healthcare Business Capstone.

This script:
1) Loads patients.csv, visits.csv, and billing.csv (or billing_clean.csv if available)
2) Recreates the SQL joins in Python
3) Engineers required features:
   - visit_frequency (per patient)
   - avg_los_per_patient
   - provider_rejection_rate
   - days_since_registration
   - visit_month, visit_dayofweek
4) Exports model_table.csv for Phase 3 modeling

Usage:
    python build_features.py --patients "F:/AI ML/capstone/patients.csv" --visits "F:/AI ML/capstone/visits.csv" --billing "F:/AI ML/capstone/billing.csv" --out model_table_features.csv
"""

import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", default="F:/AI ML/capstone/patients.csv", help="Path to patients.csv")
    parser.add_argument("--visits", default="F:/AI ML/capstone/visits.csv", help="Path to visits.csv")
    parser.add_argument("--billing", default="F:/AI ML/capstone/billing.csv", help="Path to billing.csv ")
    parser.add_argument("--out", default="model_table_feature.csv", help="Output path for modeling dataset")
    return parser.parse_args()

def main():
    args = parse_args()

    patients = pd.read_csv(args.patients)
    visits = pd.read_csv(args.visits)
    billing = pd.read_csv(args.billing)

    # Recreate SQL joins: visits INNER JOIN patients, LEFT JOIN billing
    df = (
        visits
        .merge(patients, on="patient_id", how="inner")
        .merge(billing, on="visit_id", how="left")
    )

    # ---- Feature Engineering ----

    # Visit frequency per patient
    visit_freq = df.groupby("patient_id")["visit_id"].count().rename("visit_frequency")
    df = df.merge(visit_freq, on="patient_id", how="left")

    # Average LOS per patient
    avg_los = df.groupby("patient_id")["length_of_stay_hours"].mean().rename("avg_los_per_patient")
    df = df.merge(avg_los, on="patient_id", how="left")

    # Provider rejection rate
    df["is_rejected"] = (df["claim_status"] == "Rejected").astype(int)
    provider_reject = (
        df.groupby("insurance_provider")["is_rejected"]
          .mean()
          .rename("provider_rejection_rate")
    )
    df = df.merge(provider_reject, on="insurance_provider", how="left")

    # Days since registration
    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")
    df["days_since_registration"] = (df["visit_date"] - df["registration_date"]).dt.days

    # Time-based features
    df["visit_month"] = df["visit_date"].dt.month
    df["visit_dayofweek"] = df["visit_date"].dt.dayofweek

    # Save modeling dataset
    df.drop(columns=["is_rejected"], inplace=True, errors="ignore")
    df.to_csv(args.out, index=False)

    print(f"Saved modeling dataset to: {args.out}")

if __name__ == "__main__":
    main()
