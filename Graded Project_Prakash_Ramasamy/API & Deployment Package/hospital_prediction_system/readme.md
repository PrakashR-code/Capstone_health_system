# Hospital Operations & Revenue Risk Intelligence Platform

**Author:** Prakash Ramasamy  
**Program:** AI & Machine Learning — Healthcare Analytics Capstone  
**GitHub:** https://github.com/PrakashR-code/Capstone_health_system  
**Submission Tag:** `submission-final`

---

An end-to-end healthcare analytics and machine learning system designed to help hospitals monitor operational risk and predict insurance claim outcomes.

This project integrates **SQL analytics, machine learning, explainability, drift monitoring, and FastAPI deployment** across six phases to simulate a real-world hospital intelligence platform.

---

## Project Overview

Hospitals manage complex operational workflows including patient visits, billing, insurance approvals, and revenue collection.

This system provides two predictive capabilities:

### 1. Visit Risk Prediction

Predicts whether a hospital visit is **Low / Medium / High** risk.  
Helps hospitals identify potentially critical visits earlier and allocate resources proactively.

### 2. Claim Outcome Prediction

Predicts insurance claim outcome: **Paid / Pending / Rejected**.  
Helps finance teams detect revenue risk before a claim is submitted.

---

## System Architecture

```
Hospital Data Sources
(patients.csv / visits.csv / billing.csv)

        │
        ▼
Phase 1 — SQL Analytics

        │
        ▼
Phase 2 — EDA & Feature Engineering  →  model_table.csv

        │
        ▼
Phase 3 — ML Model Training
  ├── risk_model.pkl   (sklearn Pipeline + RandomForest)
  └── claim_model.pkl  (raw RandomForestClassifier)

        │
        ▼
Phase 4 — Model Evaluation & Explainability (SHAP, Fairness)

        │
        ▼
Phase 5 — FastAPI Service (main.py + schemas.py + utils.py)
  ├── POST /predict_risk
  ├── POST /predict_claim
  └── Gradio UI (ui.py)

        │
        ▼
Phase 6 — Monitoring & Drift Detection
  ├── monitoring.py            (online feature + prediction drift)
  └── advanced_monitoring.py   (batch dataset drift)

        │
        ▼
Hospital Dashboards / Internal Applications
```

---

## Technologies Used

| Category | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn (RandomForest, LogisticRegression, GradientBoosting, SMOTE) |
| API Framework | FastAPI, Uvicorn, Pydantic |
| UI | Gradio |
| Monitoring | Custom drift detection (mean deviation + categorical shift) |
| Database / Analytics | MySQL, SQL |
| Deployment | AWS EC2, Ubuntu 22.04 |
| Version Control | Git, GitHub |

---

## Project Structure

```
hospital_prediction_system/
├── app/
│   ├── main.py                  # FastAPI entry point — endpoints, model loading, logging
│   ├── schemas.py               # Pydantic request/response models
│   ├── utils.py                 # Feature preparation, encoding, model loading
│   ├── predict_risk_updated.py  # Risk prediction helpers
│   ├── predict_claim_updated.py # Claim prediction helpers
│   └── ui.py                    # Gradio interactive UI
├── data/
│   ├── model_table.csv          # Merged training dataset (25,000 rows × 33 cols)
│   ├── risk_features.json       # Risk model feature schema
│   ├── claim_features.json      # Claim model feature schema
│   └── drift_report.csv         # Latest monitoring output
├── logs/
│   ├── risk_predictions_log.csv  # Audit log for risk predictions
│   └── claim_predictions_log.csv # Audit log for claim predictions
├── models/
│   ├── risk_model.pkl            # Trained risk model (sklearn Pipeline)
│   └── claim_model.pkl           # Trained claim model (raw RandomForest)
├── monitoring/
│   ├── monitoring.py             # Online drift detection (feature + prediction)
│   └── advanced_monitoring.py   # Batch drift detection (numeric + categorical)
├── requirements.txt
├── run.sh                        # Start API script
└── run_ui.sh                     # Start Gradio UI script
```

---

## Machine Learning Models

### Algorithms Evaluated

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline (with class_weight='balanced') |
| Decision Tree | Non-linear interpretable model |
| Random Forest | **Final selected model** |
| Gradient Boosting | Ensemble comparison |

### Final Model: Random Forest

Selected because:
- Best overall accuracy on time-based test split
- `class_weight='balanced'` handles imbalanced claim outcomes
- No feature scaling required — robust to different numeric scales
- 200 trees reduce variance from edge-case billing patterns
- Supports feature importance for explainability

### Risk Model — Feature Pipeline

Uses a full **sklearn Pipeline** with `ColumnTransformer`:
- `OneHotEncoder` for: `gender`, `department`, `visit_type`
- Passthrough for numeric: `age`, `length_of_stay_hours`, `visit_frequency`, `avg_los_per_patient`, `days_since_registration`
- `age_group` is auto-derived server-side from `age` — not required in API input

### Claim Model — Manual Encoding

Uses a **raw RandomForestClassifier** (no built-in pipeline).  
Categorical encoding is applied in `utils.py` before inference:

| Column | Encoding |
|---|---|
| gender | M→1, F→0 |
| department | factorize (Cardiology=0, Orthopedics=1, ICU=2, General=3, ER=4, Neurology=5) |
| visit_type | factorize (ER=0, OPD=1, ICU=2) |

The same encoding maps are stored in `claim_features.json` and replicated in `utils.py` to prevent train/serve skew.

---

## Model Performance

### Claim Outcome Model

| Metric | Value |
|---|---|
| Accuracy | ~55% |
| Weighted F1 | ~0.52 |

**Note on accuracy:** The model uses 10 pre-adjudication features — information available *before* a claim decision is made. The column `approved_amount` was deliberately excluded as it equals $0 for all Rejected claims and mirrors `billed_amount` for Paid claims — encoding the adjudication decision itself (target leakage). Including it inflated accuracy to ~90%. **55% is the honest ceiling** for pre-adjudication prediction on this synthetic dataset.

### Visit Risk Model

| Metric | Value |
|---|---|
| Accuracy | ~47% |
| Weighted F1 | ~0.40 |

**Note on accuracy:** The `risk_score` label in this synthetic dataset was not derived from the available patient/visit features — feature means are nearly identical across all three risk groups (High/Medium/Low). The model correctly learns the class distribution but cannot exceed random baseline without clinically meaningful features such as diagnosis codes or severity scores.

| Feature | High | Low | Medium |
|---|---|---|---|
| Mean age | 44.78 | 44.69 | 44.89 |
| Mean LOS hours | 19.76 | 19.15 | 20.08 |
| Mean visit_frequency | 5.93 | 6.00 | 5.92 |
| chronic_flag=1 ratio | 49.5% | 50.3% | 50.6% |

---

## API Endpoints

### Health Check

```
GET /health
```

Response:
```json
{ "status": "API running" }
```

---

### Risk Prediction

```
POST /predict_risk
```

Request body (10 fields — `age_group` is derived automatically):

```json
{
  "age": 60,
  "gender": "M",
  "department": "ICU",
  "visit_type": "ICU",
  "chronic_flag": 1,
  "length_of_stay_hours": 24.0,
  "visit_frequency": 5,
  "avg_los_per_patient": 18.0,
  "days_since_registration": 900,
  "billed_amount": 45000.0
}
```

Response:
```json
{ "risk_prediction": "High" }
```

---

### Claim Prediction

```
POST /predict_claim
```

Request body (10 fields):

```json
{
  "age": 45,
  "gender": "F",
  "department": "General",
  "visit_type": "OPD",
  "length_of_stay_hours": 12.5,
  "visit_frequency": 2,
  "chronic_flag": 1,
  "provider_rejection_rate": 0.05,
  "visit_intensity": 1.2,
  "billed_amount": 15000.0
}
```

Response:
```json
{ "claim_prediction": "Paid" }
```

---

## Prediction Logging

All predictions are appended to CSV audit logs automatically:

- `logs/risk_predictions_log.csv`
- `logs/claim_predictions_log.csv`

Each row captures: `timestamp`, `input features`, `prediction`, `model_version`

This enables **auditability, governance, and drift detection** over time.

---

## Local Deployment (Step by Step)

**Prerequisites:** Python 3.10+, pip

**Step 1 — Unzip and navigate:**
```bash
unzip hospital_prediction_system.zip
cd hospital_prediction_system
```

**Step 2 — Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

**Step 3 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 4 — Start the API:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# or use the provided script:
./run.sh
```

**Step 5 — Access Swagger UI:**
```
http://127.0.0.1:8000/docs
```

**Step 6 — Start the Gradio UI (separate terminal):**
```bash
source venv/bin/activate
python app/ui.py
# or:
./run_ui.sh
```
```
http://127.0.0.1:7860
```

---

## AWS EC2 Deployment

### Architecture

```
Client Applications (Dashboards / UI)
        │
        ▼
FastAPI Server (AWS EC2 — Ubuntu 22.04, t2.micro)
        │
        ▼
risk_model.pkl / claim_model.pkl
        │
        ▼
Prediction Logs (CSV audit trail)
```

### Step 1 — Launch EC2 Instance

1. Go to **AWS Console → EC2 → Launch Instance**
2. Select **Ubuntu 22.04 LTS**
3. Instance type: **t2.micro** (free tier eligible)
4. Storage: minimum **20 GB**

### Step 2 — Key Pair

1. Create new key pair → RSA → download `.pem` file
2. **Save securely** — cannot be re-downloaded

### Step 3 — Security Group

Configure inbound rules:

| Port | Protocol | Purpose |
|---|---|---|
| 22 | SSH | Remote access |
| 8000 | TCP | FastAPI |
| 7860 | TCP | Gradio UI |

### Step 4 — Connect to Instance

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<PUBLIC_IP>
```

### Step 5 — Upload Files via AWS CloudShell

1. Open **AWS CloudShell** in the console
2. Upload both `your-key.pem` and `hospital_prediction_system.zip`
3. Transfer to the EC2 instance:

```bash
chmod 400 your-key.pem
scp -i your-key.pem hospital_prediction_system.zip ubuntu@<PUBLIC_IP>:/home/ubuntu/
```

### Step 6 — Setup Environment on EC2

```bash
sudo apt update
sudo apt install python3-pip python3-venv unzip -y
unzip hospital_prediction_system.zip
cd hospital_prediction_system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 7 — Start the API

```bash
# Background process (keeps running after SSH disconnect)
./run.sh &

# Or manually:
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
```

### Step 8 — Start the Gradio UI

Open a second terminal session:

```bash
cd hospital_prediction_system
source venv/bin/activate
python app/ui.py
```

### Step 9 — Access the Deployed Service

| Service | URL |
|---|---|
| Swagger API docs | `http://<PUBLIC_IP>:8000/docs` |
| Gradio UI | `http://<PUBLIC_IP>:7860` |

---

## Monitoring & Drift Detection

### Online Monitoring (monitoring.py)

Compares live prediction logs against the training baseline:
- **Feature drift** — flags numeric features where the live mean deviates >10% from training
- **Prediction distribution** — tracks the proportion of each output label over time

```bash
python monitoring/monitoring.py
```

Output: `data/drift_report.csv`

### Batch Drift Detection (advanced_monitoring.py)

Compares a new incoming dataset against the training reference:
- **Numeric drift** — absolute mean deviation per feature
- **Categorical drift** — detects new unseen category values (can break encoding at inference)

Prepare new data file at `data/new_model_table.csv`, then run:

```bash
python monitoring/advanced_monitoring.py
```

Output:
- `data/advanced_drift_report.csv`
- `data/advanced_drift_report.json`

**Drift threshold:** 0.1 (10% absolute mean shift). Adjust `THRESHOLD` in the script to suit your tolerance.

---

## Business Impact

This system helps hospitals:

- Detect **high-risk visits** earlier for proactive resource allocation
- Identify **rejected insurance claims** before submission to reduce revenue leakage
- Improve **revenue cycle efficiency** through pre-adjudication prediction
- Support **data-driven operational decisions** backed by explainable ML
- Maintain **governance and auditability** through structured prediction logging

---

## Future Improvements

- Integrate diagnosis codes and clinical severity scores into the risk model
- Replace synthetic `risk_score` labels with clinically derived risk rules
- Add SHAP-based explainability endpoint (`/explain`)
- Dockerise the service for container-based deployment
- Connect to a live hospital database instead of CSV inputs
- Build a dashboard in Streamlit or Power BI consuming the API
- Set up AWS CloudWatch alerts triggered by drift_report findings