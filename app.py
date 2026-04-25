import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# --- Feature config ---
numeric_cols = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
categorical_cols = [
    'Lead Origin', 'Lead Source', 'Last Activity',
    'What is your current occupation', 'Country'
]
feature_cols = numeric_cols + categorical_cols
target_col = 'Converted'

# --- Train and save model ---
def train_model():
    print("Training model...")
    df_raw = pd.read_csv('Lead Scoring.csv')

    df_clean = df_raw[feature_cols + [target_col]].copy()

    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna('Unknown')

    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols)

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000))
    ])
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    lr_pipeline.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    importances = rf_model.feature_importances_
    fi = sorted(zip(X.columns.tolist(), importances), key=lambda x: -x[1])

    bundle = {
        'lr': lr_pipeline,
        'rf': rf_model,
        'feature_names': X.columns.tolist(),
        'feature_importances': fi
    }

    with open('models.pkl', 'wb') as f:
        pickle.dump(bundle, f)

    print("✅ Model trained and saved!")
    return bundle

# --- Load or train model ---
if os.path.exists('models.pkl'):
    with open('models.pkl', 'rb') as f:
        bundle = pickle.load(f)
else:
    bundle = train_model()

lr = bundle['lr']
rf = bundle['rf']
feature_names = bundle['feature_names']

# --- Load data ---
df = pd.read_csv('Lead Scoring.csv')

# --- Prediction function ---
def predict(data):
    row = {feature: 0 for feature in feature_names}

    row['TotalVisits'] = data['TotalVisits']
    row['Total Time Spent on Website'] = data['time_spent']
    row['Page Views Per Visit'] = data['page_views']

    for col, val in {
        'Lead Origin': data['lead_origin'],
        'Lead Source': data['lead_source'],
        'Last Activity': data['last_activity'],
        'What is your current occupation': data['occupation'],
        'Country': data['country'],
    }.items():
        key = f"{col}_{val}"
        if key in row:
            row[key] = 1

    X = pd.DataFrame([row])[feature_names]

    lr_prob = float(lr.predict_proba(X)[0][1])
    rf_prob = float(rf.predict_proba(X)[0][1])
    ensemble = (lr_prob + rf_prob) / 2

    return lr_prob, rf_prob, ensemble