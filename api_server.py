from flask import Flask, request, jsonify
import pickle, numpy as np, pandas as pd

app = Flask(__name__)

with open('models.pkl', 'rb') as f:
    bundle = pickle.load(f)

lr = bundle['lr']
rf = bundle['rf']
le_source = bundle['le_source']
le_company = bundle['le_company']
feature_names = bundle['feature_names']

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Build a single row dataframe
    row = {
        'TotalVisits': data.get('TotalVisits', 0),
        'Total Time Spent on Website': data.get('Total Time Spent on Website', 0),
        'Page Views Per Visit': data.get('Page Views Per Visit', 0),
    }

    # Add categorical dummies
    cat_fields = {
        'Lead Origin': data.get('Lead Origin', 'Unknown'),
        'Lead Source': data.get('Lead Source', 'Unknown'),
        'Last Activity': data.get('Last Activity', 'Unknown'),
        'What is your current occupation': data.get('occupation', 'Unknown'),
        'Country': data.get('Country', 'Unknown'),
    }
    for col, val in cat_fields.items():
        key = f"{col}_{val}"
        row[key] = 1

    # Align to training features
    X = pd.DataFrame([row]).reindex(columns=feature_names, fill_value=0)

    lr_prob = float(lr.predict_proba(X)[0][1])
    rf_prob = float(rf.predict_proba(X)[0][1])
    ensemble = (lr_prob + rf_prob) / 2

    return jsonify({
        'conversion_probability': round(ensemble, 3),
        'logistic_regression': round(lr_prob, 3),
        'random_forest': round(rf_prob, 3),
        'recommendation': 'High Priority' if ensemble > 0.5 else 'Medium' if ensemble > 0.25 else 'Low Priority'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models': ['logistic_regression', 'random_forest']})

if __name__ == '__main__':
    app.run(debug=True, port=5001)