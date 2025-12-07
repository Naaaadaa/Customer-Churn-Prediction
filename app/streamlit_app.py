import os
import pandas as pd
import streamlit as st
import joblib

# Load trained model and sample data

@st.cache_resource
def load_model():
    model_path = os.path.join("models", "churn_logreg_pipeline.pkl")
    model = joblib.load(model_path)
    return model

@st.cache_data
def load_sample_data():
    data_path = os.path.join("data", "telco_churn.csv")
    df = pd.read_csv(data_path)

    # Clean TotalCharges just like in the notebook
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    # We don't need the target for building input template
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    return df

model = load_model()
sample_df = load_sample_data()

# Use the first row as a template for expected columns
input_template = sample_df.iloc[0:1].copy()

# Get options for some categorical features from the data itself
gender_options = sorted(sample_df["gender"].unique().tolist())
internet_options = sorted(sample_df["InternetService"].unique().tolist())
contract_options = sorted(sample_df["Contract"].unique().tolist())
payment_options = sorted(sample_df["PaymentMethod"].unique().tolist())
phone_options = sorted(sample_df["PhoneService"].unique().tolist())
partner_options = sorted(sample_df["Partner"].unique().tolist())
dependents_options = sorted(sample_df["Dependents"].unique().tolist())
paperless_options = sorted(sample_df["PaperlessBilling"].unique().tolist())
onlinesec_options = sorted(sample_df["OnlineSecurity"].unique().tolist())
techsupport_options = sorted(sample_df["TechSupport"].unique().tolist())


# Streamlit App UI

st.title("📉 Telecom Customer Churn Prediction")

st.write(
    "This app uses a Logistic Regression model to predict whether a telecom customer "
    "is likely to **churn** (leave the service) based on their account and service details."
)

st.sidebar.header("Customer Information")

# ---- Basic customer info ----
gender = st.sidebar.selectbox("Gender", gender_options)
senior = st.sidebar.selectbox("Senior Citizen (0 = No, 1 = Yes)", [0, 1])
partner = st.sidebar.selectbox("Has Partner?", partner_options)
dependents = st.sidebar.selectbox("Has Dependents?", dependents_options)

tenure = st.sidebar.slider("Tenure (months with company)", min_value=0, max_value=72, value=12)

# ---- Services ----
phone_service = st.sidebar.selectbox("Phone Service", phone_options)
internet_service = st.sidebar.selectbox("Internet Service", internet_options)
online_security = st.sidebar.selectbox("Online Security", onlinesec_options)
tech_support = st.sidebar.selectbox("Tech Support", techsupport_options)

# ---- Contract & Billing ----
contract = st.sidebar.selectbox("Contract Type", contract_options)
paperless = st.sidebar.selectbox("Paperless Billing", paperless_options)
payment_method = st.sidebar.selectbox("Payment Method", payment_options)

monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=70.0, step=1.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=monthly_charges * max(tenure, 1), step=10.0)

# Build input row


# Start from template and overwrite with user choices
input_row = input_template.copy()

input_row["gender"] = gender
input_row["SeniorCitizen"] = senior
input_row["Partner"] = partner
input_row["Dependents"] = dependents
input_row["tenure"] = tenure
input_row["PhoneService"] = phone_service
input_row["InternetService"] = internet_service
input_row["OnlineSecurity"] = online_security
input_row["TechSupport"] = tech_support
input_row["Contract"] = contract
input_row["PaperlessBilling"] = paperless
input_row["PaymentMethod"] = payment_method
input_row["MonthlyCharges"] = monthly_charges
input_row["TotalCharges"] = total_charges

st.subheader("Customer Summary (Model Input)")
st.write(input_row)

if st.button("Predict Churn"):
    churn_proba = model.predict_proba(input_row)[0, 1]
    churn_pred = model.predict(input_row)[0]

    if churn_pred == 1:
        st.error(f"⚠ The model predicts this customer is likely to CHURN.\n\nEstimated churn probability: **{churn_proba:.2f}**")
    else:
        st.success(f" The model predicts this customer is likely to STAY.\n\nEstimated churn probability: **{churn_proba:.2f}**")
