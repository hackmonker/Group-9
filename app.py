import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model, scaler, and training column names
try:
    with open('rf_model.pkl', 'rb') as model_file:
        rf_model = pickle.load(model_file)

    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    with open('X_train_columns.pkl', 'rb') as col_file:
        train_cols = pickle.load(col_file)

except FileNotFoundError:
    st.error("Required files (model, scaler, or column names) not found. Please ensure 'rf_model.pkl', 'scaler.pkl', and 'X_train_columns.pkl' are in the same directory.")
    st.stop() # Stop execution if files are not found

st.title("Customer Churn Prediction")

st.write("Enter customer details to predict churn:")

# Create input fields for the features
# The preprocess_input function will handle mapping these to the trained model's features.

st.header("Customer Information")

gender_male = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 1)
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=100.0)


# Map input values to the format used for training (one-hot encoded and scaled)
def preprocess_input(gender_male, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines,
                       internet_service, online_security, online_backup, device_protection, tech_support,
                       streaming_tv, streaming_movies, contract, paperless_billing, payment_method,
                       monthly_charges, total_charges, train_cols): # Pass train_cols to the function

    # Create a dictionary to hold the input values, matching the original DataFrame structure before one-hot encoding
    input_data = {
        'gender': gender_male,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    # Create a pandas DataFrame from the input data
    df_input_raw = pd.DataFrame([input_data])

    # Apply the same transformations as in preprocessing, in the same order
    df_input_raw['SeniorCitizen'] = df_input_raw['SeniorCitizen'].apply(lambda x: 1 if x == "Yes" else 0)
    df_input_raw['TotalCharges'] = pd.to_numeric(df_input_raw['TotalCharges'], errors='coerce').fillna(df_input_raw['TotalCharges'].median()) # Use median just in case, though input is numeric


    categorical_cols_to_encode = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                                 'PaperlessBilling', 'PaymentMethod']

    df_input_processed = pd.get_dummies(df_input_raw, columns=categorical_cols_to_encode, drop_first=True)

    # Reindex to ensure all training columns are present and in the correct order
    # Use the loaded train_cols here
    df_input_reindexed = df_input_processed.reindex(columns=train_cols, fill_value=0)

    # Apply scaling
    # Identify numerical columns in the reindexed dataframe (which should match training numerical columns)
    numerical_cols_final = df_input_reindexed.select_dtypes(include=np.number).columns
    df_scaled_final = scaler.transform(df_input_reindexed[numerical_cols_final])

    # Replace the numerical columns in the reindexed dataframe with the scaled values
    df_input_reindexed[numerical_cols_final] = df_scaled_final


    return df_input_reindexed

# Predict button
if st.button("Predict Churn"):
    # Preprocess the input data, passing the loaded train_cols
    processed_input_data = preprocess_input(gender_male, senior_citizen, partner, dependents, tenure,
                                            phone_service, multiple_lines, internet_service, online_security,
                                            online_backup, device_protection, tech_support, streaming_tv,
                                            streaming_movies, contract, paperless_billing, payment_method,
                                            monthly_charges, total_charges, train_cols) # Pass train_cols

    # Make prediction
    prediction = rf_model.predict(processed_input_data)

    # Display the prediction
    if prediction[0] == 1:
        st.write("Prediction: Customer is likely to Churn")
    else:
        st.write("Prediction: Customer is unlikely to Churn")
