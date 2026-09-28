from flask import Flask, request, jsonify, send_from_directory
import joblib, numpy as np, pandas as pd, os, warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static')

BASE = os.path.dirname(__file__)
lr     = joblib.load(os.path.join(BASE, 'lr_model.pkl'))
rf     = joblib.load(os.path.join(BASE, 'rf_model.pkl'))
scaler = joblib.load(os.path.join(BASE, 'scaler.pkl'))
le     = joblib.load(os.path.join(BASE, 'label_encoder.pkl'))

NUMERICAL = ['Age','Weight_kg','Height_cm','BMI','Daily_Caloric_Intake',
             'Cholesterol_mg/dL','Blood_Pressure_mmHg','Glucose_mg/dL',
             'Weekly_Exercise_Hours','Adherence_to_Diet_Plan','Dietary_Nutrient_Imbalance_Score']

ALL_FEATURES = lr.feature_names_in_.tolist()

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    row = {f: 0 for f in ALL_FEATURES}

    # numerical
    for col in NUMERICAL:
        row[col] = float(data.get(col, 0))

    # one-hot categoricals
    def set_ohe(prefix, val):
        key = f"{prefix}_{val}"
        if key in row:
            row[key] = 1

    set_ohe('Gender', data.get('Gender','Male'))
    set_ohe('Disease_Type', data.get('Disease_Type','None'))
    set_ohe('Severity', data.get('Severity','Mild'))
    set_ohe('Physical_Activity_Level', data.get('Physical_Activity_Level','Moderate'))
    set_ohe('Dietary_Restrictions', data.get('Dietary_Restrictions','None'))
    set_ohe('Allergies', data.get('Allergies','None'))
    set_ohe('Preferred_Cuisine', data.get('Preferred_Cuisine','Chinese'))

    df = pd.DataFrame([row])
    df[NUMERICAL] = scaler.transform(df[NUMERICAL])

    lr_pred  = le.inverse_transform(lr.predict(df))[0]
    rf_pred  = le.inverse_transform(rf.predict(df))[0]
    lr_proba = lr.predict_proba(df)[0].tolist()
    rf_proba = rf.predict_proba(df)[0].tolist()
    classes  = le.classes_.tolist()

    return jsonify({
        'lr': {'prediction': lr_pred, 'probabilities': dict(zip(classes, lr_proba))},
        'rf': {'prediction': rf_pred, 'probabilities': dict(zip(classes, rf_proba))},
        'classes': classes
    })

if __name__ == '__main__':
    app.run(port=5000, debug=False)
