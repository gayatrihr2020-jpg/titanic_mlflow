import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("MLflow Learning")

with mlflow.start_run():
    mlflow.log_param("algorithm", "RandomForest")
    mlflow.log_param("n_estimators", 100)

    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("f1_score", 0.82)

    print("MLflow run completed successfully")

