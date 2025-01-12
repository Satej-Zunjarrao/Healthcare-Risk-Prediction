"""
Module: Model Deployment
Author: Satej
Description: This script deploys the trained machine learning model as a Flask API,
             allowing healthcare systems to access predictions in real-time.
"""

import joblib
from flask import Flask, request, jsonify
import pandas as pd

# Load the best model
MODEL_PATH = "satej_best_model.pkl"
model = joblib.load(MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    """
    Default route for API health check.
    """
    return jsonify({"message": "Healthcare Prediction API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Route to make predictions using the deployed model.
    Accepts JSON input with feature values and returns predictions.

    Request Body:
    {
        "features": [
            { "feature1": value1, "feature2": value2, ... }
        ]
    }

    Response:
    {
        "predictions": [0, 1, ...]
    }
    """
    try:
        # Parse input data
        input_data = request.json.get("features", [])
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert input data to DataFrame
        input_df = pd.DataFrame(input_data)
        
        # Perform prediction
        predictions = model.predict(input_df).tolist()
        
        # Return predictions
        return jsonify({"predictions": predictions})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)
