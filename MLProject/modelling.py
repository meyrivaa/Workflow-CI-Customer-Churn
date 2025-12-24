import os
import pandas as pd
import mlflow
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Inisialisasi DagsHub
dagshub.init(
    repo_owner="Meyrivaa", 
    repo_name="Eksperimen_SML_Salwa-Salsabila-Meyriva", 
    mlflow=True
)

# Autologging 
mlflow.autolog()

# Config
DATA_PATH = "ecommerce-customer-churn_dataset_preprocessing.csv"
MODEL_NAME = "customer_churn_model"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Memilih fitur (menghindari data leakage)
leakage_cols = [
    "customer_id", "signup_date", "last_purchase_date", 
    "days_since_last_purchase", "total_orders", 
    "total_spent_usd", "avg_order_value", "is_premium_member"
]

X = df.drop(columns=leakage_cols + ["churned"])
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Run MLflow
with mlflow.start_run(run_name="Basemodel_RandomForest"):
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    
    # Prediksi
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    print(f"Model Training Selesai. Accuracy: {acc}")