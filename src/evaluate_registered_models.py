import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from mlflow.models import evaluate


# ============================================================
# MLflow configuration
# ============================================================

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Titanic Registered Model Evaluation")


# ============================================================
# Load Titanic dataset
# ============================================================

df = pd.read_csv("data/titanic.csv")

target = "Survived"

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]

X = df[features]
y = df[target]


# ============================================================
# Recreate the SAME test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# MLflow evaluation expects the target column
eval_data = X_test.copy()

# MLflow evaluation works more reliably when numeric
# columns that may contain missing values are float64.
numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
]

for column in numeric_features:
    eval_data[column] = eval_data[column].astype("float64")

# Use ordinary Python/object strings rather than pandas StringDtype.
categorical_features = [
    "Sex",
    "Embarked",
]

for column in categorical_features:
    eval_data[column] = eval_data[column].astype(object)

eval_data[target] = y_test.to_numpy()


# ============================================================
# Registered models
# ============================================================

models = {
    "RandomForest_v2": "models:/TitanicSurvivalModel/2",
    "DecisionTree_v3": "models:/TitanicSurvivalModel/3",
}


# ============================================================
# Evaluate each registered model
# ============================================================

results = {}

for model_name, model_uri in models.items():

    print("\n" + "=" * 60)
    print(f"Evaluating: {model_name}")
    print(f"Model URI : {model_uri}")
    print("=" * 60)

    with mlflow.start_run(
        run_name=f"Evaluation_{model_name}"
    ) as run:

        mlflow.set_tag(
            "evaluation_type",
            "registered_model_evaluation"
        )

        mlflow.set_tag(
            "model_name",
            model_name
        )

        mlflow.set_tag(
            "model_uri",
            model_uri
        )

        mlflow.set_tag(
            "dataset",
            "Titanic"
        )

        mlflow.log_param(
            "evaluation_rows",
            len(eval_data)
        )

        # ----------------------------------------------------
        # MLflow evaluates the EXISTING registered model
        # ----------------------------------------------------

        result = evaluate(
            model=model_uri,
            data=eval_data,
            targets=target,
            model_type="classifier",
        )

        results[model_name] = result

        print("\nMetrics:")
        for metric_name, metric_value in result.metrics.items():
            print(f"{metric_name}: {metric_value}")

        print(f"\nMLflow Run ID: {run.info.run_id}")


# ============================================================
# Compare the two evaluation results
# ============================================================

print("\n\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison_metrics = [
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "roc_auc",
]

for metric in comparison_metrics:

    print(f"\n{metric}")

    for model_name, result in results.items():

        value = result.metrics.get(metric)

        if value is not None:
            print(f"  {model_name}: {value:.4f}")