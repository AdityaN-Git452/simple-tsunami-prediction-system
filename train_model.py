# train_model.py
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data_path = "earthquake_data_tsunami.csv"
df = pd.read_csv(data_path)

print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())

# --- Preprocessing (same logic from notebook) ---
# Drop missing values if any
df = df.dropna()

# Define feature columns and target variable (update these if needed)
# Assuming 'Risk' or similar is target
target_col = 'Risk' if 'Risk' in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {acc:.2f}")

# Save model as pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("💾 Model saved as model.pkl")
