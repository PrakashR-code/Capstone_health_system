from pydantic import BaseModel

# =========================
# Risk Model Request
# =========================
class RiskRequest(BaseModel):
    age: int
    gender: str
    department: str
    visit_type: str
    chronic_flag: int
    length_of_stay_hours: float
    visit_frequency: float
    avg_los_per_patient: float
    days_since_registration: int
    billed_amount: float


# =========================
# Claim Model Request
# =========================
class ClaimRequest(BaseModel):
    age: int
    gender: str
    department: str
    visit_type: str
    length_of_stay_hours: float
    visit_frequency: float
    chronic_flag: int
    provider_rejection_rate: float
    visit_intensity: float
    billed_amount: float