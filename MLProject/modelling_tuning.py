import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Menghubungkan ke DagsHub + MLflow
dagshub.init(
    repo_owner="meyrivaa",
    repo_name="Eksperimen_SML_Salwa-Salsabila-Meyriva",
    mlflow=True
)

mlflow.set_experiment("Modeling_Customer_Churn_Tuning")

# Load Dataset
data_path = "ecommerce-customer-churn_dataset_preprocessing.csv"
df = pd.read_csv(data_path)

print("Kolom dataset:")
print(df.columns)

# Mengurangi fitur yang berpotensi data leakage
df = df.drop(columns=[
    "customer_id",
    "last_purchase_date",
    "days_since_last_purchase"
])

# Feature engineering 
df["orders_bin"] = pd.cut(
    df["total_orders"],
    bins=[-1, 2, 10, 1000],
    labels=[0, 1, 2]
)

df["spending_bin"] = pd.qcut(
    df["total_spent_usd"],
    q=3,
    labels=[0, 1, 2]
)

df = df.drop(columns=[
    "total_orders",
    "total_spent_usd",
    "avg_order_value"
])

# Encoding categorical
df = pd.get_dummies(df, drop_first=True)

# Feature & Target
X = df.drop("churned", axis=1)
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Hyperparameter Tuning
param_grid = {
    "n_estimators": [150, 200],
    "max_depth": [8, 10, 12],
    "min_samples_split": [5, 10]
}

base_model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    max_features="sqrt"
)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

# Train Model
with mlflow.start_run(run_name="RandomForest-Tuning-Final"):

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

# Evaluasi Model
train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

y_test_prob = best_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_test_prob)

print("\nBest Parameters:", grid_search.best_params_)
print("Train Accuracy:", train_acc)
print("Test Accuracy :", test_acc)
print("ROC-AUC       :", roc_auc)

print("\nClassification Report:")
print(classification_report(y_test, test_pred))

# Metrik & Parameter
mlflow.log_metric("train_accuracy", train_acc)
mlflow.log_metric("test_accuracy", test_acc)
mlflow.log_metric("roc_auc", roc_auc)

for param, value in grid_search.best_params_.items():
    mlflow.log_param(param, value)

mlflow.log_param("model", "RandomForestClassifier")
mlflow.log_param("class_weight", "balanced")

# Confusion Matrix
cm = confusion_matrix(y_test, test_pred)

os.makedirs("artifacts", exist_ok=True)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Tuned Model")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")

cm_path = "artifacts/confusion_matrix_tuning.png"
plt.savefig(cm_path)
plt.close()

mlflow.log_artifact(cm_path)

# Classification Report
report_path = "artifacts/classification_report_tuning.txt"
with open(report_path, "w") as f:
    f.write(classification_report(y_test, test_pred))

mlflow.log_artifact(report_path)

# Save Model
mlflow.sklearn.log_model(best_model, "RandomForest_Tuned_Model")