from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved model and scaler
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.get_json(force=True)
        df_predict = pd.DataFrame([data])

        # Ensure the column order matches the training data used for the scaler and model
        # This is crucial for correct scaling and prediction.
        # We need the columns from X_train, which was used for fitting the scaler and model.
        # Assuming the order of columns in X_train is the expected order.
        # If the input data is missing columns, add them with default values (e.g., 0 or median)
        # If the input data has extra columns, they will be ignored by reindexing.
        # This assumes X_train is available in the environment, which it is from previous steps.

        # Get the column names from X_train
        train_cols = X_train.columns

        # Reindex the input DataFrame to match the training columns
        df_predict = df_predict.reindex(columns=train_cols, fill_value=0)

        # Apply the loaded scaler to the numerical features
        # Identify numerical columns in the input DataFrame
        numerical_cols_predict = df_predict.select_dtypes(include=np.number).columns
        df_predict[numerical_cols_predict] = scaler.transform(df_predict[numerical_cols_predict])

        # Make predictions
        prediction = rf_model.predict(df_predict)

        # Return the prediction as JSON
        # The prediction is a numpy array, convert it to a list for JSON serialization
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # For development, run with debug=True
    # In production, use a production-ready WSGI server like Gunicorn or uWSGI
    app.run(debug=True)
