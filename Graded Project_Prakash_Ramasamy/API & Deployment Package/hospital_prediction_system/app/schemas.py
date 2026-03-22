"""
schemas.py — Pydantic Request Schemas

Defines the validated input shapes for both prediction endpoints.
Pydantic automatically validates types and raises HTTP 422 on bad input.

Note: age_group is intentionally absent from RiskRequest.
      It is computed at runtime in main.py (_derive_age_group) and
      injected into the feature vector before the model is called.
"""

from pydantic import BaseModel

# =========================
# Risk Model Request
# =========================
class RiskRequest(BaseModel):
    """Input schema for the /predict_risk endpoint.

    Maps to risk_features.json. 10 user-supplied fields; age_group is derived server-side.
    """
    age: int                        # Patient age in years
    gender: str                     # 'M' or 'F'
    department: str                 # Treating department (e.g. 'ICU', 'ER')
    visit_type: str                 # Type of visit: 'ER', 'ICU', or 'OPD'
    chronic_flag: int               # 1 = patient has a chronic condition, 0 = none
    length_of_stay_hours: float     # Duration of current visit in hours
    visit_frequency: float          # Number of visits by this patient in the dataset
    avg_los_per_patient: float      # Patient's average LOS across all visits
    days_since_registration: int    # Days since the patient first registered
    billed_amount: float            # Total amount billed for this visit


# =========================
# Claim Model Request
# =========================
class ClaimRequest(BaseModel):
    """Input schema for the /predict_claim endpoint.

    Maps to claim_features.json. 10 user-supplied fields.
    Categorical fields are encoded in utils.prepare_features() before model inference.
    """
    age: int                        # Patient age in years
    gender: str                     # 'M' or 'F'
    department: str                 # Treating department
    visit_type: str                 # 'ER', 'ICU', or 'OPD'
    length_of_stay_hours: float     # Duration of visit in hours
    visit_frequency: float          # How often this patient visited
    chronic_flag: int               # 1 = chronic condition present
    provider_rejection_rate: float  # Historical rejection rate of the billing provider (0-1)
    visit_intensity: float          # Composite score of procedures / complexity
    billed_amount: float            # Total amount billed for this visit