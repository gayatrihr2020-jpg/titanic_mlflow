import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# MLflow configuration
# ============================================================

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Titanic Survival Prediction"
REGISTERED_MODEL_NAME = "TitanicSurvivalModel"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv("data/titanic.csv")


# ============================================================
# Features and target
# ============================================================

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]

target = "Survived"

X = df[features]
y = df[target]


# ============================================================
# Feature groups
# ============================================================

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
]

categorical_features = [
    "Sex",
    "Embarked",
]


# ============================================================
# Train/test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# ============================================================
# Preprocessing
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# ============================================================
# Hyperparameter search space
# ============================================================

param_grid = {
    "max_depth": [
        3,
        6,
        10,
        None,
    ],
    "min_samples_split": [
        2,
        5,
        10
    ],
    "min_samples_leaf": [
        1,
        2,
        5
    ],
    "criterion": [
        "gini",
        "entropy",
    ],
}


# ============================================================
# Create realistic MLflow signature input
# ============================================================

signature_input = X_test.copy()

for column in numeric_features:
    signature_input[column] = (
        signature_input[column].astype("float64")
    )

signature_input = signature_input.head(3)


# ============================================================
# Tuning
# ============================================================

best_f1 = -1.0
best_params = None
best_model_id = None
best_run_id = None


print("\n")
print("=" * 70)
print("DECISION TREE HYPERPARAMETER TUNING")
print("=" * 70)

print("\nSearch space:")

for parameter, values in param_grid.items():
    print(f"{parameter}: {values}")


# Number of combinations:
# 4 x 3 x 3 x 2 = 72

total_trials = (
    len(param_grid["max_depth"])
    * len(param_grid["min_samples_split"])
    * len(param_grid["min_samples_leaf"])
    * len(param_grid["criterion"])
)

print(f"\nTotal trials: {total_trials}")


# ============================================================
# Parent MLflow run
# ============================================================

with mlflow.start_run(
    run_name="DecisionTree_Hyperparameter_Tuning"
) as parent_run:

    mlflow.set_tag(
        "experiment_type",
        "hyperparameter_tuning",
    )

    mlflow.set_tag(
        "algorithm",
        "DecisionTreeClassifier",
    )

    mlflow.set_tag(
        "optimization_metric",
        "f1_score",
    )

    mlflow.log_param(
        "total_trials",
        total_trials,
    )

    mlflow.log_param(
        "test_size",
        0.2,
    )

    mlflow.log_param(
        "random_state",
        42,
    )

    # --------------------------------------------------------
    # Iterate through hyperparameters
    # --------------------------------------------------------

    trial_number = 0

    for max_depth in param_grid["max_depth"]:

        for min_samples_split in param_grid[
            "min_samples_split"
        ]:

            for min_samples_leaf in param_grid[
                "min_samples_leaf"
            ]:

                for criterion in param_grid[
                    "criterion"
                ]:

                    trial_number += 1

                    run_name = (
                        f"DT_trial_{trial_number}_"
                        f"depth_{max_depth}_"
                        f"split_{min_samples_split}_"
                        f"leaf_{min_samples_leaf}_"
                        f"{criterion}"
                    )

                    print(
                        f"\nTrial {trial_number}/{total_trials}: "
                        f"{run_name}"
                    )

                    # ------------------------------------------------
                    # Child MLflow run
                    # ------------------------------------------------

                    with mlflow.start_run(
                        run_name=run_name,
                        nested=True,
                    ) as child_run:

                        # --------------------------------------------
                        # Create classifier
                        # --------------------------------------------

                        classifier = DecisionTreeClassifier(
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            criterion=criterion,
                            random_state=42,
                        )

                        # --------------------------------------------
                        # Complete ML pipeline
                        # --------------------------------------------

                        model = Pipeline(
                            steps=[
                                (
                                    "preprocessor",
                                    preprocessor,
                                ),
                                (
                                    "classifier",
                                    classifier,
                                ),
                            ]
                        )

                        # --------------------------------------------
                        # Log parameters
                        # --------------------------------------------

                        mlflow.log_param(
                            "algorithm",
                            "DecisionTreeClassifier",
                        )

                        mlflow.log_param(
                            "max_depth",
                            max_depth,
                        )

                        mlflow.log_param(
                            "min_samples_split",
                            min_samples_split,
                        )

                        mlflow.log_param(
                            "min_samples_leaf",
                            min_samples_leaf,
                        )

                        mlflow.log_param(
                            "criterion",
                            criterion,
                        )

                        mlflow.log_param(
                            "random_state",
                            42,
                        )

                        # --------------------------------------------
                        # Train
                        # --------------------------------------------

                        model.fit(
                            X_train,
                            y_train,
                        )

                        # --------------------------------------------
                        # Predict
                        # --------------------------------------------

                        y_pred = model.predict(
                            X_test
                        )

                        # --------------------------------------------
                        # Metrics
                        # --------------------------------------------

                        accuracy = accuracy_score(
                            y_test,
                            y_pred,
                        )

                        precision = precision_score(
                            y_test,
                            y_pred,
                        )

                        recall = recall_score(
                            y_test,
                            y_pred,
                        )

                        f1 = f1_score(
                            y_test,
                            y_pred,
                        )

                        cm = confusion_matrix(
                            y_test,
                            y_pred,
                        )

                        # --------------------------------------------
                        # Log metrics
                        # --------------------------------------------

                        mlflow.log_metric(
                            "accuracy",
                            accuracy,
                        )

                        mlflow.log_metric(
                            "precision",
                            precision,
                        )

                        mlflow.log_metric(
                            "recall",
                            recall,
                        )

                        mlflow.log_metric(
                            "f1_score",
                            f1,
                        )

                        # --------------------------------------------
                        # Log confusion matrix values
                        # --------------------------------------------

                        mlflow.log_metric(
                            "true_negative",
                            int(cm[0][0]),
                        )

                        mlflow.log_metric(
                            "false_positive",
                            int(cm[0][1]),
                        )

                        mlflow.log_metric(
                            "false_negative",
                            int(cm[1][0]),
                        )

                        mlflow.log_metric(
                            "true_positive",
                            int(cm[1][1]),
                        )

                        # --------------------------------------------
                        # Model signature
                        # --------------------------------------------

                        signature_output = model.predict(
                            signature_input
                        )

                        signature = infer_signature(
                            signature_input,
                            signature_output,
                        )

                        # --------------------------------------------
                        # Log model
                        # --------------------------------------------

                        model_info = mlflow.sklearn.log_model(
                            sk_model=model,
                            name="titanic_decision_tree",
                            skops_trusted_types=[
                                "numpy.dtype",
                            ],
                            signature=signature,
                            input_example=signature_input,
                        )

                        # --------------------------------------------
                        # Print metrics
                        # --------------------------------------------

                        print(
                            f"Accuracy: {accuracy:.4f} | "
                            f"F1: {f1:.4f}"
                        )

                        # --------------------------------------------
                        # Track best model
                        # --------------------------------------------

                        if f1 > best_f1:

                            best_f1 = f1

                            best_params = {
                                "max_depth": max_depth,
                                "min_samples_split":
                                    min_samples_split,
                                "min_samples_leaf":
                                    min_samples_leaf,
                                "criterion": criterion,
                            }

                            best_model_id = (
                                model_info.model_id
                            )

                            best_run_id = (
                                child_run.info.run_id
                            )


# ============================================================
# Display best result
# ============================================================

print("\n")
print("=" * 70)
print("TUNING COMPLETED")
print("=" * 70)

print("\nBest F1 Score:")
print(f"{best_f1:.4f}")

print("\nBest Parameters:")

for parameter, value in best_params.items():
    print(f"{parameter}: {value}")

print("\nBest MLflow Run ID:")
print(best_run_id)

print("\nBest MLflow Model ID:")
print(best_model_id)


# ============================================================
# Register best model
# ============================================================

print("\n")
print("=" * 70)
print("REGISTERING BEST MODEL")
print("=" * 70)

model_uri = f"models:/{best_model_id}"

client = MlflowClient(
    tracking_uri=MLFLOW_TRACKING_URI
)

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=REGISTERED_MODEL_NAME,
)

print(
    f"\nRegistered model: "
    f"{registered_model.name}"
)

print(
    f"New model version: "
    f"{registered_model.version}"
)


# ============================================================
# Assign candidate alias
# ============================================================

client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="candidate",
    version=registered_model.version,
)

print(
    f"\nAlias 'candidate' assigned to "
    f"Version {registered_model.version}"
)


# ============================================================
# Final information
# ============================================================

print("\n")
print("=" * 70)
print("MODEL REGISTRATION COMPLETED")
print("=" * 70)

print(
    f"Model: {REGISTERED_MODEL_NAME}"
)

print(
    f"Version: {registered_model.version}"
)

print(
    f"Alias: candidate"
)

print(
    f"Best F1: {best_f1:.4f}"
)

print("\nCurrent champion was NOT changed.")