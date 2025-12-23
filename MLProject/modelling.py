import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Config
DATA_PATH = "ecommerce-customer-churn_dataset_preprocessing.csv"
MODEL_NAME = "customer_churn_model"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Kolom dataset:")
print(df.columns)

leakage_cols = [
    "customer_id",
    "signup_date",
    "last_purchase_date",
    "days_since_last_purchase",
    "total_orders",
    "total_spent_usd",
    "avg_order_value",
    "is_premium_member",
]

X = df.drop(columns=leakage_cols + ["churned"])
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=10,
    random_state=42,
    class_weight="balanced",
)

model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

mlflow.log_metric("train_accuracy", train_acc)
mlflow.log_metric("test_accuracy", test_acc)

# Artifacts
os.makedirs("artifacts", exist_ok=True)

cm = confusion_matrix(y_test, model.predict(X_test))
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d")
plt.savefig("artifacts/confusion_matrix.png")
plt.close()

mlflow.log_artifact("artifacts/confusion_matrix.png")

# Log model 
mlflow.sklearn.log_model(
    sk_model=model,
    name=MODEL_NAME
)

print(f"Model logged as '{MODEL_NAME}'")
