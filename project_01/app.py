import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

st.title('ChurnGuard: Customer Churn Predictor')

# Load models
if not os.path.exists('churn_model.pkl'):
    st.error('⚠️ Model file not found!')
    st.warning('Please run the training script first:')
    st.code('python train_models.py', language='bash')
    st.info('This will train and save the churn prediction models.')
    st.stop()

try:
    with open('churn_model.pkl', 'rb') as f:
        models = pickle.load(f)
    preprocessor = models['preprocessor']
    stack_model = models['stack_model']
    rev_model = models['rev_model']
    arima_fit = models['arima_fit']
except Exception as e:
    st.error(f'Error loading models: {e}')
    st.stop()


st.write('Upload customer data (CSV matching Telco format) or enter single customer details.')

# Option 1: Single prediction
with st.form(key='single'):
    # Inputs for key features (add all as needed)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    tenure = st.number_input('Tenure (months)', min_value=0)
    monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0)
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    # ... Add more inputs for all features

    submit = st.form_submit_button('Predict Churn')
    if submit:
        input_df = pd.DataFrame({
            'gender': [gender], 'tenure': [tenure], 'MonthlyCharges': [monthly_charges], 'Contract': [contract],
            # Fill all columns
        })
        input_prep = preprocessor.transform(input_df)
        prob = stack_model.predict_proba(input_prep)[0][1]
        st.write(f'Churn Probability: {prob:.2%}')
        loss = rev_model.predict([[monthly_charges, tenure]])[0]
        st.write(f'Estimated Revenue Loss if Churns: ${loss:.2f}')

# Option 2: Batch upload
uploaded_file = st.file_uploader('Upload CSV for batch prediction')
if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    batch_prep = preprocessor.transform(batch_df)
    batch_df['Churn_Prob'] = stack_model.predict_proba(batch_prep)[:,1]
    st.dataframe(batch_df)
    # Add forecast visualization
    forecast = arima_fit.forecast(steps=12)
    st.line_chart(forecast)

# Run: streamlit run app.py