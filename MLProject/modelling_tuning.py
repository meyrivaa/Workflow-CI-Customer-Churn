import pandas as pd
import dagshub
import mlflow
import os
from sklearn.model_selection import train_test_split, GridSearchCV
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

mlflow.set_experiment("Modeling_Customer_Churn_Tuning")

# Load Data
DATA_PATH = "ecommerce-customer-churn_dataset_preprocessing.csv"
df = pd.read_csv(DATA_PATH)

# Menghapus kolom yang berpotensi data leakage 
leakage_cols = [
    "customer_id", "signup_date", "last_purchase_date", 
    "days_since_last_purchase", "total_orders", 
    "total_spent_usd", "avg_order_value", "is_premium_member"
]
X = df.drop(columns=leakage_cols + ["churned"])
y = df["churned"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Proses tuning dengan MLflow
with mlflow.start_run(run_name="Tuning_RandomForest_Final"):
    # Parameter yang diuji
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5]
    }
    
    # Inisialisasi Model
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    
    # GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Hasil terbaik
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    print(f"Tuning Selesai!")
    print(f"Parameter Terbaik: {grid_search.best_params_}")
    print(f"Akurasi Model Terbaik: {acc}")
