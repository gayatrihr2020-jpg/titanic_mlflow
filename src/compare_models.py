import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# MLflow configuration
# ============================================================

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("Titanic Survival Prediction")


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
# Preprocessing builders
# ============================================================

def create_preprocessor(use_scaling=False):
    """
    Create preprocessing pipeline.

    Numeric features:
        - Median imputation
        - Optional standard scaling

    Categorical features:
        - Most-frequent imputation
        - One-hot encoding
    """

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(strategy="median"),
        )
    ]

    if use_scaling:
        numeric_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
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
# Algorithms
# ============================================================

models = {
    "LogisticRegression": {
        "model": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "use_scaling": True,
    },

    "DecisionTree": {
        "model": DecisionTreeClassifier(
            random_state=42,
        ),
        "use_scaling": False,
    },

    "RandomForest": {
        "model": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "use_scaling": False,
    },

    "GradientBoosting": {
        "model": GradientBoostingClassifier(
            random_state=42,
        ),
        "use_scaling": False,
    },

    "KNN": {
        "model": KNeighborsClassifier(
            n_neighbors=5,
        ),
        "use_scaling": True,
    },
}


# ============================================================
# Experiment tracking
# ============================================================

results = []


for model_name, model_config in models.items():

    classifier = model_config["model"]
    use_scaling = model_config["use_scaling"]

    print("\n" + "=" * 60)
    print(f"Training: {model_name}")
    print("=" * 60)

    # --------------------------------------------------------
    # Create preprocessing pipeline
    # --------------------------------------------------------

    preprocessor = create_preprocessor(
        use_scaling=use_scaling
    )

    # --------------------------------------------------------
    # Create complete ML pipeline
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=model_name
    ):

        # ----------------------------------------------------
        # Tags
        # ----------------------------------------------------

        mlflow.set_tag(
            "experiment_type",
            "algorithm_comparison",
        )

        mlflow.set_tag(
            "algorithm",
            model_name,
        )

        mlflow.set_tag(
            "preprocessing_scaling",
            str(use_scaling),
        )

        # ----------------------------------------------------
        # Common parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "test_size",
            0.2,
        )

        mlflow.log_param(
            "random_state",
            42,
        )

        # ----------------------------------------------------
        # Model-specific parameters
        # ----------------------------------------------------

        if model_name == "LogisticRegression":

            mlflow.log_param(
                "max_iter",
                1000,
            )

        elif model_name == "DecisionTree":

            mlflow.log_param(
                "criterion",
                classifier.criterion,
            )

        elif model_name == "RandomForest":

            mlflow.log_param(
                "n_estimators",
                classifier.n_estimators,
            )

        elif model_name == "GradientBoosting":

            mlflow.log_param(
                "n_estimators",
                classifier.n_estimators,
            )

        elif model_name == "KNN":

            mlflow.log_param(
                "n_neighbors",
                classifier.n_neighbors,
            )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test
        )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Create realistic input example
        # ----------------------------------------------------
        #
        # Convert integer columns that may contain missing
        # values to float. This avoids MLflow's schema warning.
        #

        signature_input = X_test.copy()

        for column in numeric_features:
            signature_input[column] = (
                signature_input[column]
                .astype("float64")
            )

        signature_input = signature_input.head(3)

        signature_output = model.predict(
            signature_input
        )

        # ----------------------------------------------------
        # Model signature
        # ----------------------------------------------------

        signature = infer_signature(
            signature_input,
            signature_output,
        )

        # ----------------------------------------------------
        # Log model
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            name=f"titanic_{model_name.lower()}",
            skops_trusted_types=[
                "numpy.dtype",
                (
                    "sklearn.metrics."
                    "_dist_metrics.EuclideanDistance64"
                ),
                (
                    "sklearn.neighbors."
                    "_kd_tree.KDTree"
                ),
            ],
            signature=signature,
            input_example=signature_input,
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append(
            {
                "algorithm": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print(
            f"Scaling used   : {use_scaling}"
        )

        print(
            f"Accuracy       : {accuracy:.4f}"
        )

        print(
            f"Precision      : {precision:.4f}"
        )

        print(
            f"Recall         : {recall:.4f}"
        )

        print(
            f"F1 Score       : {f1:.4f}"
        )


# ============================================================
# Comparison summary
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="f1_score",
    ascending=False,
)


print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print("\n")
print("Comparison metric: F1 Score")

print("\nAlgorithm with highest F1 score:")
print(
    results_df.iloc[0]["algorithm"]
)

print(
    f"F1 Score: "
    f"{results_df.iloc[0]['f1_score']:.4f}"
)