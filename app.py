import pandas as pd
import pickle

# --- Load model ---
with open('models.pkl', 'rb') as f:
    bundle = pickle.load(f)

lr = bundle['lr']
rf = bundle['rf']
feature_names = bundle['feature_names']

# --- Load data ---
df = pd.read_csv('Lead Scoring.csv')

# --- Prediction function ---
def predict(data):
    row = {
        'TotalVisits': data['TotalVisits'],
        'Total Time Spent on Website': data['time_spent'],
        'Page Views Per Visit': data['page_views'],
    }
    for col, val in {
        'Lead Origin': data['lead_origin'],
        'Lead Source': data['lead_source'],
        'Last Activity': data['last_activity'],
        'What is your current occupation': data['occupation'],
        'Country': data['country'],
    }.items():
        row[f"{col}_{val}"] = 1

    X = pd.DataFrame([row]).reindex(columns=feature_names, fill_value=0)
    lr_prob = float(lr.predict_proba(X)[0][1])
    rf_prob = float(rf.predict_proba(X)[0][1])
    ensemble = (lr_prob + rf_prob) / 2
    return lr_prob, rf_prob, ensemble