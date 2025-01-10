import pickle
import os
from api.schemas import PredictInput

# Load the model once during service initialization
MODEL_PATH = os.getenv("MODEL_PATH", "models/data/processed/model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def predict(input_data: PredictInput) -> str:
    """
    Predicts the class of an Iris flower based on input features.

    Args:
        input_data (PredictInput): Features of the flower.
    Returns:
        str: Predicted class.
    """
    features = [
        [
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width,
        ]
    ]
    prediction = model.predict(features)[0]
    return prediction
