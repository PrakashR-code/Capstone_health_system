
from pydantic import BaseModel

class RiskRequest(BaseModel):
    age: int
    chronic_flag: int
    length_of_stay_hours: float
    visit_frequency: float
    avg_los_per_patient: float
    provider_rejection_rate: float
    days_since_registration: int
    visit_month: int
    visit_dayofweek: int
    department: str
    visit_type: str

class ClaimRequest(BaseModel):
    billed_amount: float
    approved_amount: float
    payment_days: int
    department: str
    visit_type: str
