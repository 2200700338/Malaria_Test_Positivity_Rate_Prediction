from flask import Flask, render_template, request
import pandas as pd
import joblib
from scipy.special import expit

# =========================
# 1. Initialize Flask
# =========================
app = Flask(__name__)

# =========================
# 2. Load Model & Scaler
# =========================
arimax_model = joblib.load("arimax_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# 3. Define Predictors (must match training)
# =========================
predictor_cols = [
    "STOMACH ACHE", "FEVER", "HEAD ACHE", "NAUSEA/VOMITING",
    "MALAISE", "DIARRHOEA", "WEEKLY TEMPERATURE",
    "WEEKLY RAINFALL", "WEEKLY HUMIDITY"
]


# =========================
# 4. Home Page
# =========================
@app.route('/')
def home():
    return render_template('index.html', predictors=predictor_cols)


# =========================
# 5. Prediction Route
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read input from form in the same order as predictor_cols
        input_data = [float(request.form[col]) for col in predictor_cols]

        # Convert to DataFrame
        exog_df = pd.DataFrame([input_data], columns=predictor_cols)

        # Scale using saved scaler
        exog_scaled = scaler.transform(exog_df)

        # Single-step forecast
        forecast_logit = arimax_model.get_forecast(steps=1, exog=exog_scaled).predicted_mean
        forecast = expit(forecast_logit)  # ensures prediction is between 0 and 1

        # Round & convert to percentage
        result = round(forecast.iloc[0] * 100, 2)

        return render_template('index.html', predictors=predictor_cols, prediction=result)

    except Exception as e:
        return f"Error in prediction: {e}"


# =========================
# 6. Run App
# =========================
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
