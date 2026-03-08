# 🏥 Hospital Operations & Revenue Risk Intelligence Platform

An end-to-end healthcare analytics and machine learning system designed to help hospitals monitor operational risk and predict insurance claim outcomes.

This project integrates **SQL analytics, machine learning, explainability, and FastAPI deployment** to simulate a real-world hospital intelligence platform.

---

# 📊 Project Overview

Hospitals manage complex operational workflows including patient visits, billing, insurance approvals, and revenue collection.

This system provides two predictive capabilities:

## 1️⃣ Visit Risk Prediction

Predicts whether a hospital visit is:

- Low Risk
- Medium Risk
- High Risk

Helps hospitals identify potentially critical visits earlier.

## 2️⃣ Claim Outcome Prediction

Predicts insurance claim outcome:

- Paid
- Pending
- Rejected

Helps finance teams detect **revenue risk before claim submission**.

---

# 🧠 System Architecture

```
Hospital Data Sources
(Patients / Visits / Billing CSV)

        │
        ▼
SQL Analytics (Phase 1)

        │
        ▼
EDA & Feature Engineering (Phase 2)

        │
        ▼
Machine Learning Models (Phase 3)

        │
        ▼
Model Evaluation & Explainability (Phase 4)

        │
        ▼
FastAPI Deployment (Phase 5)

        │
        ▼
Hospital Dashboards / Applications
```

---

# ⚙️ Technologies Used

### Data Processing
- Python
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- Random Forest
- Gradient Boosting
- Logistic Regression
- Decision Tree

### API Deployment
- FastAPI
- Uvicorn
- Pydantic

### Visualization / Interface
- Gradio

### Database / Analytics
- MySQL
- SQL

---

# 📂 Project Structure

```
Capstone_health_system/

phase1/
    SQL analytics queries

phase2/
    EDA notebook
    Feature engineering script
    Data quality report

phase3/
    Risk model training
    Claim model training
    Model comparison
    risk_model.pkl
    claim_model.pkl

phase4/
    Model evaluation
    Explainability
    Fairness analysis

phase5/
    main.py
    schemas.py
    utils.py
    predict_risk.py
    ui.py
    requirements.txt

reports/
    Risk_Model_Evaluation_Report.docx
    Claim_Model_Evaluation_Report.docx
    Model_Card.docx
    Explainability_Summary.docx
```

---

# 🤖 Machine Learning Models

Multiple algorithms were evaluated:

| Model | Purpose |
|------|------|
| Logistic Regression | Baseline |
| Decision Tree | Non-linear model |
| Random Forest | Final selected model |
| Gradient Boosting | Ensemble comparison |

## Final Model Selection

**Random Forest** was selected because:

- Highest overall accuracy
- Balanced performance across classes
- Strong recall for **Rejected claims**
- Supports feature importance analysis

---

# 📈 Model Performance

## Claim Outcome Model

| Metric | Value |
|------|------|
| Accuracy | ~93% |
| Macro F1 Score | ~0.91 |
| Rejected Claim Recall | ~0.96 |

This allows hospitals to **identify revenue risk early**.

---

## Visit Risk Model

Performance is moderate due to limited operational features.

Future improvements may include:

- Diagnosis codes
- Clinical severity scores
- Patient history features

---

# 🚀 FastAPI Deployment

The models are deployed as a **real-time API service**.

## Health Check

```
GET /health
```

Response:

```json
{
 "status": "API running"
}
```

---

## Risk Prediction

```
POST /predict_risk
```

Example request:

```json
{
 "age": 60,
 "chronic_flag": 1,
 "length_of_stay_hours": 24,
 "visit_frequency": 5,
 "avg_los_per_patient": 18,
 "provider_rejection_rate": 0.2,
 "days_since_registration": 900,
 "visit_month": 7,
 "visit_dayofweek": 3,
 "department": "ICU",
 "visit_type": "ICU"
}
```

---

## Claim Prediction

```
POST /predict_claim
```

Example request:

```json
{
 "billed_amount": 15000,
 "approved_amount": 12000,
 "payment_days": 25,
 "department": "General",
 "visit_type": "OPD"
}
```

---

# 🧾 Prediction Logging

Each prediction logs:

- timestamp
- model version
- feature hash

Example:

```
timestamp,model,feature_hash
2026-03-08,risk_model,ab39fd92...
```

This enables **auditability and governance**.

---

# 🖥 Running the API Locally

Install dependencies:

```
pip install -r requirements.txt
```

Start API:

```
uvicorn main:app --reload
```

Access API documentation:

```
http://127.0.0.1:8000/docs
```

---

# ☁️ Deployment

The system can be deployed on:

- AWS EC2
- Docker containers
- Kubernetes
- Cloud API gateways

---

# 📊 Business Impact

This system helps hospitals:

✔ Detect **high-risk visits** earlier  
✔ Identify **rejected insurance claims** before submission  
✔ Improve **revenue cycle efficiency**  
✔ Support **data-driven operational decisions**

---

# 👨‍💻 Author

**Prakash R**

AI & Machine Learning Program  
Healthcare Analytics Capstone Project

GitHub  
https://github.com/PrakashR-code

---

# ⭐ Future Improvements

- SHAP explainability
- Real-time monitoring
- Model drift detection
- Docker deployment
- Dashboard integration