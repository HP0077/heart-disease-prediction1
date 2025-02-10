from flask import Flask, render_template, request, jsonify, url_for
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
except Exception as e:
    print("Error loading model or scaler:", e)
    model = None  # Prevent crash if model is missing
    scaler = None

@app.route("/")
def home():
    """Render the landing page"""
    return render_template("index.html")

@app.route("/predict-page")
def predict_page():
    """Render the prediction page"""
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Process user input and return prediction result"""
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()

        if not data:
            return jsonify({"error": "No data received"}), 400

        # Convert input data to NumPy array for model prediction
        input_features = np.array([[float(data.get(k, 0)) for k in [
            "male", "age", "education", "currentSmoker", "cigsPerDay",
            "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
            "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
        ]]])

        if model is None or scaler is None:
            return jsonify({"error": "Model or Scaler not loaded. Check server logs."}), 500

        # Apply feature scaling
        input_features = scaler.transform(input_features)

        # Predict risk probability
        prob = model.predict_proba(input_features)[0][1]  # Probability of high risk
        prediction = "High risk" if prob >= 0.3 else "Low risk"

        return jsonify({"prediction": prediction, "risk_probability": f"{prob:.2%}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
