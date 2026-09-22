import mlflow
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# 1. MLflow configuration
# ============================================================

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("Titanic Hybrid Logging")


# ============================================================
# 2. Enable autologging
# ============================================================

mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
)


# ============================================================
# 3. Load dataset
# ============================================================

df = pd.read_csv("data/titanic.csv")


# ============================================================
# 4. Features and target
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

X = df[features]
y = df["Survived"]


# ============================================================
# 5. Train/test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# ============================================================
# 6. Preprocessing
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

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ]
)


# ============================================================
# 7. Model
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                random_state=42
            ),
        ),
    ]
)


# ============================================================
# 8. Train
# ============================================================

with mlflow.start_run(
    run_name="DecisionTree_Hybrid"
):

    model.fit(X_train, y_train)

    # ========================================================
    # AUTOMATIC LOGGING
    # ========================================================
    #
    # MLflow sklearn autologging automatically captures
    # supported model parameters, model information,
    # signatures, input examples, and artifacts.
    #
    # ========================================================


    # ========================================================
    # MANUAL LOGGING
    # ========================================================

    # Project metadata
    mlflow.set_tag(
        "project",
        "Titanic Survival Prediction"
    )

    mlflow.set_tag(
        "logging_type",
        "autologging_plus_manual"
    )

    mlflow.set_tag(
        "model_type",
        "DecisionTree"
    )


    # Dataset information
    mlflow.log_param(
        "dataset_name",
        "Titanic"
    )

    mlflow.log_param(
        "training_rows",
        len(X_train)
    )

    mlflow.log_param(
        "test_rows",
        len(X_test)
    )

    mlflow.log_param(
        "feature_count",
        len(features)
    )


    # Predictions
    y_pred = model.predict(X_test)


    # Custom metrics
    test_accuracy = accuracy_score(
        y_test,
        y_pred
    )

    test_f1 = f1_score(
        y_test,
        y_pred
    )

    mlflow.log_metric(
        "custom_test_accuracy",
        test_accuracy
    )

    mlflow.log_metric(
        "custom_test_f1",
        test_f1
    )


    print("Hybrid logging completed.")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1: {test_f1:.4f}")