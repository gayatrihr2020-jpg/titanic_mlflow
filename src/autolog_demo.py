import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# MLflow configuration
# ============================================================

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("Titanic Autologging")


# ============================================================
# Enable MLflow autologging
# ============================================================

mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
)


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
# Model
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                criterion="entropy",
                random_state=42,
            ),
        ),
    ]
)


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
# Training
# ============================================================

with mlflow.start_run(
    run_name="DecisionTree_Autologging"
):

    model.fit(
        X_train,
        y_train,
    )

    print("\nTraining completed.")

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )