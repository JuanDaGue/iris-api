"""
Módulo: API de Predicción con FastAPI

Este módulo define un endpoint para realizar predicciones basadas en las características de entrada proporcionadas. 
El modelo predice el nombre de una planta a partir de los datos ingresados por el usuario.

Dependencias:
- fastapi
- api.schemas: Define los esquemas de entrada y salida para el endpoint.
- api.services: Contiene la lógica para realizar la predicción.

Uso:
Integre este módulo dentro de una aplicación FastAPI más amplia para habilitar predicciones basadas en el modelo.
"""

from fastapi import APIRouter, HTTPException
from api.schemas import PredictInput, PredictOutput
from api.services import predict

router = APIRouter()


@router.post("/predict", response_model=PredictOutput)
async def predict_endpoint(input_data: PredictInput):
    """
    Endpoint para realizar predicciones.

    Este endpoint recibe las características de las plantas en el formato
    especificado
    por el esquema `PredictInput` y devuelve el nombre de la planta predicha
    basado
    en el modelo entrenado.

    Argumentos:
        input_data (PredictInput): Datos de entrada que contienen las
        características de la planta.

    Retorna:
        dict: Diccionario que contiene la predicción en el formato especificado
        por `PredictOutput`.

    Excepciones:
        HTTPException: Se genera si ocurre un error durante la predicción.
    """
    try:
        prediction = predict(input_data)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Error en la predicción: {str(e)}")
