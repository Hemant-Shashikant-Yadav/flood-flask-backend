
from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Load pre-trained model and scaler
rf = joblib.load("random_forest_model.pkl")  # Replace with your model file path
scaler = joblib.load("scaler.pkl")  # Replace with your scaler file path

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.json
        input_data = pd.DataFrame([data])
        
        # Scale input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = rf.predict(input_scaled)
        result = {"FloodOccurred": int(prediction[0])}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment or default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)
