import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

# --- 1. Load real data ---
df = pd.read_csv('Lead Scoring.csv')
print(f"Dataset shape: {df.shape}")

# --- 2. Select useful features ---
feature_cols = [
    'TotalVisits',
    'Total Time Spent on Website',
    'Page Views Per Visit',
    'Lead Origin',
    'Lead Source',
    'Last Activity',
    'What is your current occupation',
    'Country'
]
target_col = 'Converted'

numeric_cols = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
categorical_cols = ['Lead Origin', 'Lead Source', 'Last Activity',
                    'What is your current occupation', 'Country']

# --- 3. Clean data ---
df = df[feature_cols + [target_col]].copy()

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')

# --- 4. Encode categorical columns ---
df = pd.get_dummies(df, columns=categorical_cols)

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 5. Train models ---
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

lr_pipeline.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# --- 6. Evaluate ---
lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]
rf_proba = rf_model.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression ===")
print(f"ROC AUC: {roc_auc_score(y_test, lr_proba):.3f}")
print(classification_report(y_test, lr_pipeline.predict(X_test)))

print("=== Random Forest ===")
print(f"ROC AUC: {roc_auc_score(y_test, rf_proba):.3f}")
print(classification_report(y_test, rf_model.predict(X_test)))

# --- 7. Feature importance ---
importances = rf_model.feature_importances_
fi = sorted(zip(feature_names, importances), key=lambda x: -x[1])
print("\n=== Top 10 Feature Importances (RF) ===")
for name, imp in fi[:10]:
    print(f"  {name}: {imp:.3f}")

# --- 8. Save ---
with open('models.pkl', 'wb') as f:
    pickle.dump({
        'lr': lr_pipeline,
        'rf': rf_model,
        'feature_names': feature_names,
        'feature_importances': fi
    }, f)