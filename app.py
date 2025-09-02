from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load saved ARIMAX model
arimax_model = joblib.load("arimax_model.pkl")

# Load scaler if needed for preprocessing
# scaler = joblib.load("scaler.pkl")  # optional if scaling ML inputs

# Define the predictors
predictor_cols = [
    "STOMACH ACHE", "FEVER", "HEAD ACHE", "NAUSEA/VOMITING",
    "DIARRHOEA", "MALAISE", "WEEKLY TEMPERATURE",
    "WEEKLY HUMIDITY", "WEEKLY RAINFALL"
]


# Route for home page (input form)
@app.route('/')
def home():
    return render_template('index.html', predictors=predictor_cols)


# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read values from form
        input_data = [float(request.form[col]) for col in predictor_cols]
        exog_df = pd.DataFrame([input_data], columns=predictor_cols)

        # Predict using ARIMAX
        pred = arimax_model.predict(start=0, end=0, exog=exog_df)
        result = round(pred.iloc[0], 2)
        return render_template('index.html', predictors=predictor_cols, prediction=result)

    except Exception as e:
        return f"Error in prediction: {e}"


if __name__ == "__main__":
    app.run(debug=True)
