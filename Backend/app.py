from flask import Flask, request, jsonify
from flask_cors import CORS  # <--- NEW LINE 1
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # <--- NEW LINE 2

# ... (keep the rest of your app.py code exactly the same below this)

# Direct path to your new 'brain' file
MODEL_PATH = r'C:\dia_predict_ai\Model\model.pkl'

# Load the model
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # The AI needs exactly 8 pieces of data to work
        features = np.array([[
            data['Pregnancies'],
            data['Glucose'],
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