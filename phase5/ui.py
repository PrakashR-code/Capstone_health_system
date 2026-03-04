import gradio as gr
import requests

RISK_API = "http://127.0.0.1:8000/predict_risk"
CLAIM_API = "http://127.0.0.1:8000/predict_claim"


# ---------------------------
# Risk Prediction
# ---------------------------

def predict_risk(
    age,
    chronic_flag,
    length_of_stay_hours,
    visit_frequency,
    avg_los_per_patient,
    provider_rejection_rate,
    days_since_registration,
    visit_month,
    visit_dayofweek,
    department,
    visit_type
):

    payload = {
        "age": int(age),
        "chronic_flag": int(chronic_flag),
        "length_of_stay_hours": float(length_of_stay_hours),
        "visit_frequency": float(visit_frequency),
        "avg_los_per_patient": float(avg_los_per_patient),
        "provider_rejection_rate": float(provider_rejection_rate),
        "days_since_registration": int(days_since_registration),
        "visit_month": int(visit_month),
        "visit_dayofweek": int(visit_dayofweek),
        "department": department,
        "visit_type": visit_type
    }

    r = requests.post(RISK_API, json=payload)

    return r.json()["prediction"]


# ---------------------------
# Claim Prediction
# ---------------------------

def predict_claim(
    billed_amount,
    approved_amount,
    payment_days,
    department,
    visit_type
):

    payload = {
        "billed_amount": float(billed_amount),
        "approved_amount": float(approved_amount),
        "payment_days": int(payment_days),
        "department": department,
        "visit_type": visit_type
    }

    r = requests.post(CLAIM_API, json=payload)

    return r.json()["prediction"]


# ---------------------------
# UI Layout
# ---------------------------

with gr.Blocks() as demo:

    gr.Markdown("# 🏥 Hospital Prediction System")

    # Risk Tab
    with gr.Tab("Risk Prediction"):

        age = gr.Textbox(label="Age")
        chronic_flag = gr.Textbox(label="Chronic Flag")
        length_of_stay_hours = gr.Textbox(label="Length of Stay Hours")
        visit_frequency = gr.Textbox(label="Visit Frequency")
        avg_los_per_patient = gr.Textbox(label="Avg LOS per Patient")
        provider_rejection_rate = gr.Textbox(label="Provider Rejection Rate")
        days_since_registration = gr.Textbox(label="Days Since Registration")
        visit_month = gr.Textbox(label="Visit Month")
        visit_dayofweek = gr.Textbox(label="Visit Day of Week")

        department = gr.Dropdown(
            ["ER", "General", "ICU", "Neurology", "Orthopedics"],
            label="Department"
        )

        visit_type = gr.Dropdown(
            ["OPD", "ICU"],
            label="Visit Type"
        )

        risk_btn = gr.Button("Predict Risk")
        risk_output = gr.Textbox(label="Risk Prediction")

        risk_btn.click(
            predict_risk,
            inputs=[
                age,
                chronic_flag,
                length_of_stay_hours,
                visit_frequency,
                avg_los_per_patient,
                provider_rejection_rate,
                days_since_registration,
                visit_month,
                visit_dayofweek,
                department,
                visit_type
            ],
            outputs=risk_output
        )

    # Claim Tab
    with gr.Tab("Claim Prediction"):

        billed_amount = gr.Textbox(label="Billed Amount")
        approved_amount = gr.Textbox(label="Approved Amount")
        payment_days = gr.Textbox(label="Payment Days")

        department2 = gr.Dropdown(
            ["ER", "General", "ICU", "Neurology", "Orthopedics"],
            label="Department"
        )

        visit_type2 = gr.Dropdown(
            ["OPD", "ICU"],
            label="Visit Type"
        )

        claim_btn = gr.Button("Predict Claim")
        claim_output = gr.Textbox(label="Claim Prediction")

        claim_btn.click(
            predict_claim,
            inputs=[
                billed_amount,
                approved_amount,
                payment_days,
                department2,
                visit_type2
            ],
            outputs=claim_output
        )

print("Starting Gradio UI...")
demo.launch(server_name="127.0.0.1", server_port=7860)