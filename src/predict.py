import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")

MODEL_URI = "models:/TitanicSurvivalModel@champion"

# Load registered model
model = mlflow.pyfunc.load_model(MODEL_URI)

# Example passenger
passenger = pd.DataFrame(
    [
        {
            "Pclass": 3,
            "Sex": "male",
            "Age": 22.0,
            "SibSp": 1,
            "Parch": 0,
            "Fare": 7.25,
            "Embarked": "S",
        }
    ]
)

# Make prediction
prediction = model.predict(passenger)

print("Model loaded successfully.")
print(f"Model URI: {MODEL_URI}")
print(f"Prediction: {prediction[0]}")