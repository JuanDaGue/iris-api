"""
Módulo: Servicio de Predicción

Este módulo implementa la lógica para cargar el modelo entrenado y realizar
predicciones basadas en las características de entrada proporcionadas.
El modelo predice el tipo de planta del conjunto de datos Iris.

Dependencias:
- os
- pickle
- api.schemas: Define el esquema de entrada para las predicciones.

Uso:
Importe este módulo y llame a la función `predict` pasando los datos de entrada
en el formato definido por `PredictInput` para obtener la predicción.
"""

import os
import pickle
from api.schemas import PredictInput

# Cargar el modelo durante la inicialización del servicio
MODEL_PATH = os.getenv("MODEL_PATH", "models/data/processed/model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Mapeo de valores objetivo a nombres de especies
CLASS_MAPPING = {0: "setosa", 1: "versicolor", 2: "virginica"}


def predict(input_data: PredictInput) -> str:
    """
    Predice el tipo de planta basado en las características de entrada.

    Este método utiliza un modelo preentrenado para predecir el nombre
    de la especie
    de una planta basándose en las características proporcionadas.

    Argumentos:
        input_data (PredictInput): Características de la planta.

    Retorna:
        str: Nombre de la especie predicha ('setosa',
        'versicolor' o 'virginica').
    """
    features = [
        [
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width,
        ]
    ]
    # Predice el valor objetivo (0, 1 o 2)
    prediction = model.predict(features)[0]
    # Mapea el valor objetivo al nombre correspondiente de la especie
    species_name = CLASS_MAPPING.get(prediction, "Unknown")
    return species_name
