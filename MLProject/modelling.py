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
EXPERIMENT_NAME = "Modeling_Customer_Churn"
DATA_PATH = "ecommerce-customer-churn_dataset_preprocessing.csv"
MODEL_ARTIFACT_PATH = "RandomForest_Baseline_Model"

mlflow.set_experiment(EXPERIMENT_NAME)


# Load Data
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("Kolom dataset:")
print(df.columns)


# Feature & Target
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
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train & Log
with mlflow.start_run(run_name="RandomForest-Baseline"):

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=10,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("Train Accuracy:", train_acc)
    print("Test Accuracy :", test_acc)

    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))

    # Log Prams & Metrics
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 8)
    mlflow.log_param("min_samples_split", 10)
    mlflow.log_param("class_weight", "balanced")

    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)

    # Artifacts
    os.makedirs("artifacts", exist_ok=True)

    cm = confusion_matrix(y_test, test_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Baseline Model")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    cm_path = "artifacts/confusion_matrix_baseline.png"
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)

    report_path = "artifacts/classification_report_baseline.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(y_test, test_pred))

    mlflow.log_artifact(report_path)

    # Log Model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=MODEL_ARTIFACT_PATH,
    )

    print(f"Model logged to MLflow as '{MODEL_ARTIFACT_PATH}'")
