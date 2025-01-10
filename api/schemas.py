"""
Módulo: Modelos de Datos para Predicción

Este módulo define los modelos de datos utilizados en la API de predicción.
Los modelos están basados en Pydantic y proporcionan validación automática
de los datos de entrada y estructura consistente para los datos de salida.

Dependencias:
- pydantic: Biblioteca para la validación de datos basada en Python.

Uso:
Importe estos modelos para definir las características de entrada esperadas
para las solicitudes de predicción y el formato de la respuesta.
"""

from pydantic import BaseModel


class PredictInput(BaseModel):
    """
    Modelo de entrada para la predicción.

    Este modelo define las características de entrada necesarias para realizar
    una predicción basada en el modelo entrenado.

    Atributos:
        sepal_length (float): Longitud del sépalo de la planta.
        sepal_width (float): Anchura del sépalo de la planta.
        petal_length (float): Longitud del pétalo de la planta.
        petal_width (float): Anchura del pétalo de la planta.
    """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictOutput(BaseModel):
    """
    Modelo de salida para la predicción.

    Este modelo define el formato de la respuesta de la API de predicción.

    Atributos:
        prediction (str): Nombre de la planta predicha por el modelo.
    """
    prediction: str
