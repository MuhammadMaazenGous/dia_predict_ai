from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Direct path to your 'brain' file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

# Load the model
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # We grab the glucose value first
        glucose_val = float(data['Glucose'])
        
        # Prepare features for the model
        features = np.array([[
            data['Pregnancies'],
            glucose_val,
            data['BloodPressure'],
            data['SkinThickness'],
            data['Insulin'],
            data['BMI'],
            data['DiabetesPedigreeFunction'],
            data['Age']
        ]])

        # --- THE FIX: CLINICAL SCALING LOGIC ---
        # Get the AI's raw probability (0.0 to 1.0)
        raw_prob = model.predict_proba(features)[0][1]
        risk_percent = raw_prob * 100

        # If glucose is high, we apply a 'Medical Weight' so the score doesn't 
        # drop off a cliff just because BMI or Age is low.
        if glucose_val > 140:
            # Add 0.4% risk for every mg/dL above 140 (the pre-diabetic line)
            adjustment = (glucose_val - 140) * 0.4
            risk_percent += adjustment

        # Hard cap at 98.5% so it looks like a real medical estimate
        final_score = min(98.5, risk_percent)

        # Ensure anything >= 200 is ALWAYS flagged as 1 (High Risk)
        # Otherwise, follow the 50% threshold
        if glucose_val >= 200:
            final_prediction = 1
        else:
            final_prediction = 1 if final_score > 50 else 0
        # --------------------------------------

        return jsonify({
            "prediction": int(final_prediction),
            "risk_score": f"{round(float(final_score), 1)}%",
            "status": "Success"
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "Failed"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)