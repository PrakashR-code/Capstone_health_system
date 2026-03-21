import gradio as gr
import requests

RISK_API = "http://127.0.0.1:8000/predict_risk"
CLAIM_API = "http://127.0.0.1:8000/predict_claim"

def safe_call(api, payload):
    r = requests.post(api, json=payload)
    try:
        return r.json()["prediction"]
    except:
        return f"API Error: {r.text}"



def predict_risk(age, gender, department, visit_type, chronic_flag, length_of_stay_hours, visit_frequency, avg_los_per_patient, days_since_registration, billed_amount):
    payload = {
        "age": int(age),
        "gender": gender,
        "department": department,
        "visit_type": visit_type,
        "chronic_flag": int(chronic_flag),
        "length_of_stay_hours": float(length_of_stay_hours),
        "visit_frequency": float(visit_frequency),
        "avg_los_per_patient": float(avg_los_per_patient),
        "days_since_registration": int(days_since_registration),
        "billed_amount": float(billed_amount)
    }
    return safe_call(RISK_API, payload)



# UPDATED CLAIM FUNCTION
def predict_claim(age, gender, dept, vtype, los, freq, chronic_flag, rejection_rate, visit_intensity_val, billed_amount):
    payload = {
        "age": int(age),
        "gender": gender,
        "department": dept,
        "visit_type": vtype,
        "length_of_stay_hours": float(los),
        "visit_frequency": float(freq),
        "chronic_flag": int(chronic_flag),
        "provider_rejection_rate": float(rejection_rate),
        "visit_intensity": float(visit_intensity_val),
        "billed_amount": float(billed_amount)
    }
    return safe_call(CLAIM_API, payload)


with gr.Blocks() as demo:
    gr.Markdown("# 🏥 Hospital Prediction System")


    # =========================
    # Risk Tab (schema-aligned)
    # =========================
    with gr.Tab("Risk Prediction"):
        age = gr.Textbox(label="Age")
        gender = gr.Dropdown(["F", "M"], label="Gender")
        department = gr.Dropdown(["Cardiology", "ER", "General", "ICU", "Neurology", "Orthopedics"], label="Department")
        visit_type = gr.Dropdown(["ER", "ICU", "OPD"], label="Visit Type")
        chronic_flag = gr.Textbox(label="Chronic Flag")
        length_of_stay_hours = gr.Textbox(label="Length of Stay (hours)")
        visit_frequency = gr.Textbox(label="Visit Frequency")
        avg_los_per_patient = gr.Textbox(label="Avg LOS per Patient")
        days_since_registration = gr.Textbox(label="Days Since Registration")
        billed_amount = gr.Textbox(label="Billed Amount")
        risk_btn = gr.Button("Predict Risk")
        risk_output = gr.Textbox(label="Risk Prediction")
        risk_btn.click(
            predict_risk,
            inputs=[age, gender, department, visit_type, chronic_flag, length_of_stay_hours, visit_frequency, avg_los_per_patient, days_since_registration, billed_amount],
            outputs=risk_output
        )


    # (Removed duplicate Risk Prediction tab)

    # =========================
    # Claim Tab (fixed indentation)
    # =========================
    with gr.Tab("Claim Prediction"):
        age2 = gr.Textbox(label="Age")
        gender2 = gr.Dropdown(["F", "M"], label="Gender")
        dept2 = gr.Dropdown(["Cardiology", "ER", "General", "ICU", "Neurology", "Orthopedics"], label="Department")
        vtype2 = gr.Dropdown(["ER", "ICU", "OPD"], label="Visit Type")
        los2 = gr.Textbox(label="Length of Stay (hours)")
        freq2 = gr.Textbox(label="Visit Frequency")
        chronic_flag2 = gr.Textbox(label="Chronic Flag")
        rejection_rate2 = gr.Textbox(label="Provider Rejection Rate")
        visit_intensity2 = gr.Textbox(label="Visit Intensity")
        billed_amount2 = gr.Textbox(label="Billed Amount")
        claim_btn = gr.Button("Predict Claim")
        out2 = gr.Textbox(label="Claim Prediction")
        claim_btn.click(
            predict_claim,
            inputs=[age2, gender2, dept2, vtype2, los2, freq2, chronic_flag2, rejection_rate2, visit_intensity2, billed_amount2],
            outputs=out2
        )

demo.launch(server_name="0.0.0.0", server_port=7860)