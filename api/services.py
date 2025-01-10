import os
import pickle
from api.schemas import PredictInput

# Load the model once during service initialization
MODEL_PATH = os.getenv("MODEL_PATH", "models/data/processed/model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Mapping from target numbers to species names
CLASS_MAPPING = {0: "setosa", 1: "versicolor", 2: "virginica"}


def predict(input_data: PredictInput) -> str:
    """
    Predice el tipo de plata basado en las caracteristicas de entrada 
    (argumento de entrada), retona el tipo de planta predicha ('setosa',
    'versicolor', or 'virginica')
    .

    Args:
        input_data (PredictInput): Features of the flower.

    Returns:
        str: Predicted species name ('setosa', 'versicolor', or 'virginica').
    """
    features = [
        [
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width,
        ]
    ]
    # predice el target value (0, 1, or 2)
    prediction = model.predict(features)[0]
    # Mapea el target value en el correspondiente nombre de la especie
    species_name = CLASS_MAPPING.get(prediction, "Unknown")
    return species_name
