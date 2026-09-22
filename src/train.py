import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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


# --------------------------------------------------
# 1. MLflow configuration
# --------------------------------------------------

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Titanic Survival Prediction")


# --------------------------------------------------
# 2. Model hyperparameters
# --------------------------------------------------

N_ESTIMATORS = 100
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
RANDOM_STATE = 42
TEST_SIZE = 0.2


# --------------------------------------------------
# 3. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/titanic.csv")


# --------------------------------------------------
# 4. Define features and target
# --------------------------------------------------

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


# --------------------------------------------------
# 5. Define feature types
# --------------------------------------------------

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


# --------------------------------------------------
# 6. Preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)


# --------------------------------------------------
# 7. Model
# --------------------------------------------------

classifier = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    random_state=RANDOM_STATE,
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ]
)


# --------------------------------------------------
# 8. Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


# --------------------------------------------------
# 9. MLflow run
# --------------------------------------------------

with mlflow.start_run():

    # Log hyperparameters
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("min_samples_split", MIN_SAMPLES_SPLIT)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("test_size", TEST_SIZE)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Model signature
    signature = infer_signature(X_test, y_pred)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        name="titanic_random_forest",
        skops_trusted_types=["numpy.dtype"],
        signature=signature,
        input_example=X_test.head(3),
    )

    # Display results
    print("\nModel Configuration:")
    print(f"Algorithm          : RandomForestClassifier")
    print(f"n_estimators       : {N_ESTIMATORS}")
    print(f"max_depth          : {MAX_DEPTH}")
    print(f"min_samples_split  : {MIN_SAMPLES_SPLIT}")

    print("\nModel Evaluation:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

print("\nTraining completed.")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")