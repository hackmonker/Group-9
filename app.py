import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model and scaler
try:
    with open('rf_model.pkl', 'rb') as model_file:
        rf_model = pickle.load(model_file)

    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'rf_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop() # Stop execution if files are not found

st.title("Customer Churn Prediction")

st.write("Enter customer details to predict churn:")

# Create input fields for the features
# You'll need to create input fields for all 30 features that were used for training.
# This requires knowledge of the columns after one-hot encoding.
# For simplicity and demonstration, let's create a few example input fields.
# You will need to expand this to include all relevant features.
# A better approach would be to load the training columns and dynamically create inputs.

# Example Input Fields (replace with actual features from your preprocessed data)
# Based on X_train.head() output, some columns are:
# SeniorCitizen, tenure, MonthlyCharges, TotalCharges, gender_Male, Partner_Yes, etc.

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
# This mapping needs to be precise to match the columns of X_train
def preprocess_input(gender_male, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines,
                       internet_service, online_security, online_backup, device_protection, tech_support,
                       streaming_tv, streaming_movies, contract, paperless_billing, payment_method,
                       monthly_charges, total_charges):

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
    df_input = pd.DataFrame([input_data])

    # Apply the same one-hot encoding as used during training
    # Need to make sure all possible categories are present in the input DataFrame columns
    # before one-hot encoding to ensure consistent column order.
    # This is tricky without access to the original categorical column values.
    # A robust way is to have a list of all possible values for each categorical column.

    # For demonstration, let's assume the categories and apply one-hot encoding.
    # You will need to adjust this based on your actual data's categories.
    categorical_cols_app = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                            'PaperlessBilling', 'PaymentMethod']

    # Manually add columns for all possible categories if they are not in the input
    # This is crucial for consistent one-hot encoding structure.
    # Example: for 'gender', add 'gender_Male' and 'gender_Female' (though with drop_first=True, only one is needed)
    # This part requires knowing all unique values from the training data for each categorical column.
    # A simpler approach for demonstration:
    df_input = pd.get_dummies(df_input, columns=categorical_cols_app, drop_first=True)

    # Ensure the columns of the input DataFrame match the columns of the training data (X_train)
    # This is vital for the scaler and the model.
    # Get the training column names from the global X_train variable
    global X_train
    train_cols = X_train.columns.tolist() # Convert to list for easier manipulation

    # Add missing columns to the input DataFrame with a default value (e.g., 0)
    for col in train_cols:
        if col not in df_input.columns:
            df_input[col] = 0

    # Reindex the input DataFrame to match the order of the training columns
    df_input = df_input.reindex(columns=train_cols, fill_value=0)


    # Apply the loaded scaler to the numerical features
    numerical_cols_predict = df_input.select_dtypes(include=np.number).columns
    df_scaled = scaler.transform(df_input[numerical_cols_predict])

    # The scaler returns a numpy array, convert it back to a DataFrame with correct columns
    df_scaled = pd.DataFrame(df_scaled, columns=numerical_cols_predict, index=df_input.index)

    # Combine the scaled numerical features with the one-hot encoded categorical features
    # Need to handle the non-numerical columns that were already one-hot encoded.
    # Let's re-create the structure to match X_train completely.

    # The easiest way is to create a DataFrame with all X_train columns and fill it.
    processed_input = pd.DataFrame(0, index=[0], columns=X_train.columns)

    # Fill the processed_input DataFrame with values from the user input,
    # applying one-hot encoding logic manually or using get_dummies then reindexing.
    # Given the complexity of dynamic mapping, let's rely on get_dummies and reindexing as above,
    # ensuring reindex fills missing columns with 0.

    # Re-doing the processing with a clear structure
    input_data_raw = {
        'gender': gender_male, 'SeniorCitizen': senior_citizen, 'Partner': partner,
        'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone_service,
        'MultipleLines': multiple_lines, 'InternetService': internet_service,
        'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
        'DeviceProtection': device_protection, 'TechSupport': tech_support,
        'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
        'Contract': contract, 'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    df_input_raw = pd.DataFrame([input_data_raw])

    # Apply the same transformations as in preprocessing, in the same order
    df_input_raw['SeniorCitizen'] = df_input_raw['SeniorCitizen'].apply(lambda x: 1 if x == "Yes" else 0)
    df_input_raw['TotalCharges'] = pd.to_numeric(df_input_raw['TotalCharges'], errors='coerce').fillna(df_input_raw['TotalCharges'].median()) # Use median just in case, though input is numeric

    categorical_cols_to_encode = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                                 'PaperlessBilling', 'PaymentMethod']

    df_input_processed = pd.get_dummies(df_input_raw, columns=categorical_cols_to_encode, drop_first=True)

    # Reindex to ensure all training columns are present and in the correct order
    df_input_reindexed = df_input_processed.reindex(columns=X_train.columns, fill_value=0)

    # Apply scaling
    # Identify numerical columns in the reindexed dataframe (which should match X_train numerical columns)
    numerical_cols_final = df_input_reindexed.select_dtypes(include=np.number).columns
    df_scaled_final = scaler.transform(df_input_reindexed[numerical_cols_final])

    # Replace the numerical columns in the reindexed dataframe with the scaled values
    df_input_reindexed[numerical_cols_final] = df_scaled_final


    return df_input_reindexed

# Predict button
if st.button("Predict Churn"):
    # Preprocess the input data
    processed_input_data = preprocess_input(gender_male, senior_citizen, partner, dependents, tenure,
                                            phone_service, multiple_lines, internet_service, online_security,
                                            online_backup, device_protection, tech_support, streaming_tv,
                                            streaming_movies, contract, paperless_billing, payment_method,
                                            monthly_charges, total_charges)

    # Make prediction
    prediction = rf_model.predict(processed_input_data)

    # Display the prediction
    if prediction[0] == 1:
        st.write("Prediction: Customer is likely to Churn")
    else:
        st.write("Prediction: Customer is unlikely to Churn")
