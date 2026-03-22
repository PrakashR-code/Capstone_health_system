# 🏥 Hospital Operations & Revenue Risk Intelligence Platform

An end-to-end healthcare analytics and machine learning system that helps hospitals monitor operational risk and predict insurance claim outcomes — built across six phases from raw SQL analytics through live API deployment and continuous monitoring.

This project integrates **SQL analytics, exploratory data analysis, machine learning, model evaluation, FastAPI deployment, and drift monitoring** to simulate a production-grade hospital intelligence platform.

> **Submission Tag**: `submission-final` | Commit: `d6faa53` | GitHub: https://github.com/PrakashR-code/Capstone_health_system

---

# 📊 Project Overview

Hospitals manage complex operational workflows — patient visits, billing, insurance approvals, and revenue collection. Delays and errors in any of these create significant financial and clinical risk.

This platform ingests raw hospital data across **25,000 visits** totalling **$521.8M in billed charges** and provides two AI-powered predictions plus continuous monitoring.

## Dataset at a Glance

| Metric | Value |
|---|---|
| Total visits analyzed | 25,000 |
| Total billed amount | $521.8M |
| Total approved amount | $387.2M |
| Rejected claims (revenue) | $74.9M (15.2%) |
| Pending claims (revenue) | $127.7M (25.1%) |
| Departments covered | General, ER, Neurology, Orthopedics, Cardiology, ICU |
| Insurance providers | MediCareX, CareOne, HealthPlus, SecureLife |
| Chronic patients | 50.3% |
| Average length of stay | 19.6 hours |

---

## Predictive Capabilities

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

# 🧠 System Architecture — 6-Phase Design

```
Raw Hospital Data  (patients.csv, visits.csv, billing.csv)
        │
        ▼
Phase 1 ── SQL Analytics & Business Intelligence
        │   KPIs: rejection rates, LOS, revenue by department
        │
        ▼
Phase 2 ── EDA & Feature Engineering
        │   25,000-row model_table built with engineered features
        │
        ▼
Phase 3 ── Machine Learning Model Training
        │   Risk model (Random Forest) + Claim model (Random Forest)
        │
        ▼
Phase 4 ── Model Evaluation & Explainability
        │   Classification reports, confusion matrices, SHAP
        │
        ▼
Phase 5 ── FastAPI Deployment + Gradio UI
        │   POST /predict_risk  |  POST /predict_claim
        │
        ▼
Phase 6 ── Drift Monitoring & Alerting
            Feature drift detection, prediction drift, alert logs
```

---

# ⚙️ Technologies Used

| Category | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn, Random Forest, Gradient Boosting, Logistic Regression |
| Balancing | SMOTE (imbalanced-learn) |
| API Deployment | FastAPI, Uvicorn, Pydantic |
| UI | Gradio |
| Database / Analytics | MySQL, SQL |
| Monitoring | Custom drift detection (PSI, KL divergence) |
| Reporting | python-docx |
| Versioning | Git, GitHub |

---

# 📂 Repository Structure

```
Capstone_health_system/
│
├── readme.md                          ← This file
│
├── Final/                             ← Graded notebooks (Phase 2–4)
│   ├── Phase2_EDA.ipynb
│   ├── Phase3_claim_model.ipynb
│   ├── Phase3_risk_model.ipynb
│   ├── Phase4_Evaluation.ipynb
│   └── data/
│       ├── model_table.csv            ← 25,000-row feature table
│       ├── patients.csv
│       ├── visits.csv
│       ├── billing.csv
│       ├── claim_features.json
│       └── risk_features.json
│
├── phase1/                            ← Raw source data
│   ├── patients.csv
│   ├── visits.csv
│   └── billing.csv
│
├── phase2/                            ← EDA & feature engineering
│   ├── 01_eda.ipynb
│   ├── build_features.py
│   └── model_table.csv
│
├── phase3/                            ← Model training
│   ├── final_phase3_risk_model.ipynb
│   ├── final_phase3_risk_model_with_smote.ipynb
│   ├── risk_features.json
│   └── claim_features.json
│
├── phase4/                            ← Evaluation & explainability
│   └── Phase4_Model_Evaluation_Final_v1.ipynb
│
├── phase5/
│   └── hospital_prediction_system/    ← Production API package
│       ├── main.py                    ← FastAPI app entry point
│       ├── schemas.py                 ← Pydantic request/response models
│       ├── utils.py                   ← Preprocessing helpers
│       ├── predict_risk.py            ← Risk prediction logic
│       ├── ui.py                      ← Gradio web interface
│       ├── requirements.txt
│       ├── models/
│       │   ├── risk_model.pkl
│       │   └── claim_model.pkl
│       ├── data/
│       │   └── model_table.csv
│       ├── logs/
│       │   └── predictions.log
│       └── monitoring/
│           ├── monitoring.py          ← PSI-based drift detection
│           └── advanced_monitoring.py ← KL divergence + alert system
│
├── phase6/                            ← Monitoring outputs
│
└── Graded Project_Prakash_Ramasamy/   ← Full submission package
    ├── API & Deployment Package/
    │   ├── Healthcare_Insights_Report.docx
    │   └── Deployment_Runbook.docx
    └── ...
```

---

# 🤖 Machine Learning Models

## Algorithm Comparison

| Algorithm | Claim Accuracy | Risk Accuracy | Notes |
|---|---|---|---|
| Logistic Regression | ~52% | ~44% | Baseline |
| Decision Tree | ~53% | ~45% | Overfits on small depth |
| Gradient Boosting | ~54% | ~46% | Marginal gain |
| **Random Forest** | **~55%** | **~47%** | **Selected — best balance** |

**Random Forest** was selected because it produced the best overall accuracy, balanced performance across all classes, and supports feature importance analysis.

---

## Feature Sets

### Risk Model — 9 Input Features

| Feature | Type | Description |
|---|---|---|
| `age` | numeric | Patient age in years |
| `chronic_flag` | binary | 1 = has chronic condition |
| `length_of_stay_hours` | numeric | Duration of visit in hours |
| `visit_frequency` | numeric | Number of visits by patient |
| `avg_los_per_patient` | numeric | Patient's average LOS across visits |
| `provider_rejection_rate` | numeric | Fraction of provider's claims rejected |
| `days_since_registration` | numeric | Days since patient first registered |
| `department` | categorical | Ward (encoded: General=0, ER=1, Neurology=2, Orthopedics=3, Cardiology=4, ICU=5) |
| `visit_type` | categorical | Type (encoded: Emergency=0, Follow-up=1, ICU=2, OPD=3, Routine=4) |

### Claim Model — 10 Input Features

| Feature | Type | Description |
|---|---|---|
| `age` | numeric | Patient age in years |
| `chronic_flag` | binary | 1 = has chronic condition |
| `length_of_stay_hours` | numeric | Duration of visit in hours |
| `visit_frequency` | numeric | Number of visits by patient |
| `avg_los_per_patient` | numeric | Patient's average LOS across visits |
| `provider_rejection_rate` | numeric | Fraction of provider's claims rejected |
| `days_since_registration` | numeric | Days since patient first registered |
| `billed_amount` | numeric | Total charge billed to insurer |
| `department` | categorical | Ward (same encoding as above) |
| `visit_type` | categorical | Type (same encoding as above) |

> **Note**: `approved_amount` and `payment_days` are NOT input features — they are post-decision values unavailable at prediction time and would create target leakage if included.

---

# 📈 Model Performance — Honest Assessment

## Claim Outcome Model (~55% Accuracy)

| Metric | Value |
|---|---|
| Accuracy | ~55% |
| Macro F1 Score | ~0.47 |
| Rejected Claim Recall | ~0.50 |

**Why ~55% and not higher?**

Early experiments reached ~93% accuracy by accidentally including `approved_amount` as a feature. For all **Rejected** claims, `approved_amount = $0` — making the target perfectly predictable from that field alone. This is **target leakage**, not a genuine model.

After removing `approved_amount` (and `payment_days`, which is similarly post-decision), accuracy settled at ~55% — an honest reflection of what the model can genuinely predict from pre-visit and visit-time features. This is the correct value.

---

## Visit Risk Model (~47% Accuracy)

| Metric | Value |
|---|---|
| Accuracy | ~47% |
| Macro F1 Score | ~0.44 |

**Why ~47%?**

The `risk_score` labels (Low / Medium / High) in the synthetic training data are not meaningfully correlated with the available features. Across all three risk groups, feature means are nearly identical (age ~44.7, LOS ~19.5h, visit frequency ~5.97). The Random Forest cannot learn separating patterns that do not exist in the label assignments.

This is a **synthetic data limitation**, not an architectural failure. With real clinical data (diagnosis codes, lab values, vital signs), accuracy would improve substantially.

---

# 🚀 FastAPI Deployment

The models are served as a real-time REST API via FastAPI.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict_risk` | Visit risk prediction |
| POST | `/predict_claim` | Claim outcome prediction |
| GET | `/docs` | Swagger UI (auto-generated) |

---

## Health Check

```
GET /health
```

Response:

```json
{
  "status": "API running",
  "version": "1.0.0"
}
```

---

## Risk Prediction

```
POST /predict_risk
```

Request body — 10 fields:

```json
{
  "age": 60,
  "chronic_flag": 1,
  "length_of_stay_hours": 24.0,
  "visit_frequency": 5,
  "avg_los_per_patient": 18.5,
  "provider_rejection_rate": 0.20,
  "days_since_registration": 900,
  "department": "ICU",
  "visit_type": "Emergency",
  "gender": "M"
}
```

Response:

```json
{
  "prediction": "High",
  "confidence": 0.72,
  "model_version": "risk_model_v1"
}
```

---

## Claim Prediction

```
POST /predict_claim
```

Request body — 10 fields:

```json
{
  "age": 45,
  "chronic_flag": 1,
  "length_of_stay_hours": 12.5,
  "visit_frequency": 2,
  "avg_los_per_patient": 10.0,
  "provider_rejection_rate": 0.05,
  "days_since_registration": 365,
  "billed_amount": 15000.0,
  "department": "General",
  "visit_type": "OPD"
}
```

Response:

```json
{
  "prediction": "Paid",
  "confidence": 0.61,
  "model_version": "claim_model_v1"
}
```

> **Important**: Do NOT include `approved_amount` or `payment_days` — these are post-decision values unavailable at prediction time and will cause a validation error.

---

# 🧾 Prediction Logging & Auditability

Every prediction call is logged automatically:

```
timestamp,model,visit_id,prediction,confidence,feature_hash
2026-03-08T10:22:31,risk_model_v1,V-10423,High,0.72,ab39fd92...
2026-03-08T10:22:35,claim_model_v1,B-10423,Paid,0.61,c9d1ea34...
```

The `feature_hash` enables reproducibility checks and audit trail compliance.

---

# 🖥 Running the API Locally

**Step 1 — Clone the repository**

```bash
git clone https://github.com/PrakashR-code/Capstone_health_system.git
cd Capstone_health_system/phase5/hospital_prediction_system
```

**Step 2 — Create a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/macOS
```

**Step 3 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4 — Start the API**

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Step 5 — Open Swagger UI**

```
http://127.0.0.1:8000/docs
```

**Step 6 — Launch the Gradio UI** (optional)

```bash
python ui.py
```

Default port: `http://127.0.0.1:7860`

---

# ☁️ AWS Deployment

## Production Deployment on AWS EC2

**Step 1 — Launch EC2 instance**
- AMI: Amazon Linux 2023
- Instance type: `t3.medium` (minimum; `t3.large` recommended for load)
- Security group: Open port 22 (SSH), 8000 (API), 80 (optional reverse proxy)

**Step 2 — Connect and install dependencies**

```bash
ssh -i your-key.pem ec2-user@<EC2-PUBLIC-IP>
sudo yum update -y
sudo yum install python3 python3-pip git -y
```

**Step 3 — Clone and set up the project**

```bash
git clone https://github.com/PrakashR-code/Capstone_health_system.git
cd Capstone_health_system/phase5/hospital_prediction_system
pip3 install -r requirements.txt
```

**Step 4 — Start the API as a background service**

```bash
nohup uvicorn main:app --host 0.0.0.0 --port 8000 &
```

**Step 5 — Verify the deployment**

```bash
curl http://<EC2-PUBLIC-IP>:8000/health
```

Expected response: `{"status": "API running", "version": "1.0.0"}`

**Step 6 — Configure systemd for auto-restart** (recommended for production)

```bash
sudo nano /etc/systemd/system/hospital-api.service
```

```ini
[Unit]
Description=Hospital Prediction API
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/Capstone_health_system/phase5/hospital_prediction_system
ExecStart=/usr/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable hospital-api
sudo systemctl start hospital-api
```

**Step 7 — Monitor logs**

```bash
sudo journalctl -u hospital-api -f
```

**Step 8 — Set up health check monitoring**

Configure AWS CloudWatch to ping `/health` every 60 seconds and alert on non-200 responses.

**Step 9 — Estimated AWS costs**

| Resource | Monthly Cost |
|---|---|
| t3.medium EC2 | ~$30 |
| 20 GB EBS storage | ~$2 |
| Data transfer (light) | ~$1–5 |
| **Total estimate** | **~$33–37/month** |

---

# 📡 Drift Monitoring

The `monitoring/` directory contains two monitoring scripts that run independently of the API.

## monitoring.py — Population Stability Index

Detects **feature drift** by comparing current data distribution against the training baseline using PSI (Population Stability Index):

- PSI < 0.1 → Stable
- PSI 0.1–0.25 → Moderate drift (investigate)
- PSI > 0.25 → Significant drift (retrain recommended)

Monitors: `age`, `billed_amount`, `length_of_stay_hours`, `visit_frequency`

```bash
python monitoring/monitoring.py
```

Output: `phase6/drift_report.csv`

---

## advanced_monitoring.py — KL Divergence + Prediction Drift

Extends basic monitoring with:
- **KL divergence** for continuous feature distributions
- **Prediction drift detection** — flags if the proportion of High Risk or Rejected predictions shifts significantly
- **Alert logging** — writes drift alerts to `phase6/alerts.log`

```bash
python monitoring/advanced_monitoring.py
```

---

# 📊 Business Impact

This system enables hospital finance and operations teams to act before problems become costly:

| Challenge | This System's Response |
|---|---|
| $74.9M in rejected claims | Predict rejection risk before submission → fix coding errors upfront |
| $127.7M in pending claims | Identify high-pending departments and insurers for follow-up prioritisation |
| 15.2% overall rejection rate | Benchmark by insurer and department; target worst performers |
| ICU & Cardiology cost overruns | Flag high-risk visits early for care coordination |
| Revenue cycle blind spots | Real-time API integrates with EHR / billing systems |

**Projected Impact** (conservative 10% improvement in rejection handling):
- Recovery potential: ~$7.5M annually
- Reduction in manual claim appeals: ~30%

---

# 👨‍💻 Author

**Prakash R**

AI & Machine Learning Program — Healthcare Analytics Capstone

GitHub: https://github.com/PrakashR-code

---

# ⭐ Future Improvements

- **SHAP explainability** — per-prediction feature importance shown in API response
- **Real diagnosis codes** — ICD-10 codes as features for substantially higher accuracy
- **Docker containerisation** — `docker-compose` for one-command deployment
- **Dashboard integration** — Power BI / Streamlit dashboard connected to prediction logs
- **Real-time streaming** — Kafka integration for continuous visit ingestion
- **Model retraining pipeline** — automated retraining triggered by high PSI alerts
- **Multi-hospital support** — tenant-aware API with per-hospital model versioning