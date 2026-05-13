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
        
        # We grab the glucose value first to check our safety rule
        glucose_val = float(data['Glucose'])
        
        # --- THE ONLY CHANGE: MEDICAL GUARDRAIL ---
        if glucose_val >= 200:
            return jsonify({
                "prediction": 1,  
                "risk_score": "95.0%", 
                "status": "Success",
                "message": "Clinical Override: High Glucose detected."
            })
        # ------------------------------------------

        # REST OF YOUR CODE REMAINS EXACTLY THE SAME
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

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "risk_score": f"{round(float(probability) * 100, 2)}%",
            "status": "Success"
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "Failed"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)