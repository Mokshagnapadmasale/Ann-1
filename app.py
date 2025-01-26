import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


with open("preprocess.pkl", "rb") as file:
    preprocess = pickle.load(file)
with open("label.pkl","rb") as file:
    label = pickle.load(file)    
model = tf.keras.models.load_model("model.h5")

st.title("Loan Application Prediction")


Age = st.slider("Age", 18.0, 90.0,value=30.0)
Gender = st.selectbox("Gender", ["male", "female"])
Education = st.selectbox("Education", ["Bachelor", "Associate", "High School", "Master", "Doctorate"])
Income = st.number_input("Income", min_value=1000, step=5000, value=50000)
Experience = st.number_input("Experience (in years)", min_value=0, step=1, value=5)
Ownership = st.selectbox("Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
Loan_amount = st.number_input("Loan Amount", min_value=1000, step=1000, value=20000)
Loan_intent = st.selectbox("Loan Intention", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL"])
Intrest = st.slider("Interest Rate (%)", 1.0, 60.0, value=15.0)
LoanIncome = st.slider("Loan to Income Ratio",0.0,1.0,step=0.1)
HistoryLength = st.slider("Credit History Length (in years)", 1.0, 35.0, value=10.0)
CreditScore = st.number_input("Credit Score", min_value=100, step=10, value=700)
PreviousLoan = st.selectbox("Previous Loan Taken?", ["Yes", "No"])


PreviousLoan_encoded = label.transform([PreviousLoan])[0]

input_data = pd.DataFrame({
    "Age": [Age],
    "Gender": [Gender],
    "Education": [Education],
    "Income": [Income],
    "Experience": [Experience],
    "Ownership": [Ownership],
    "Loan_amount": [Loan_amount],
    "Loan_intent": [Loan_intent],
    "Intrest": [Intrest],
    "LoanIncome": [LoanIncome],
    "HistoryLength": [HistoryLength],
    "CreditScore": [CreditScore],
    "PreviousLoan": [PreviousLoan_encoded]
})


st.write("Input Data:", input_data)


if st.button("Predict"):
    try:
        
        scaled_input_data = preprocess.transform(input_data)
        
        
        y_pred = model.predict(scaled_input_data)
        proba = y_pred[0][0]
        
        
        if proba > 0.5:
            st.success("Loan will be Granted")
        else:
            st.error("Loan will NOT be Granted")
            
        
        st.write(f"Prediction Probability: {proba:.2f}")
    except Exception as e:
        st.error(f"Error Occurred during prediction: {e}")
