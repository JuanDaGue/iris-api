from fastapi import APIRouter, HTTPException
from api.schemas import PredictInput, PredictOutput
from api.services import predict

router = APIRouter()


@router.post("/predict", response_model=PredictOutput)
async def predict_endpoint(input_data: PredictInput):
    """
    Enpoind predict, recive las caracteristicas de las plantas,
    y devuelve el nombre de la planta.
    """
    try:
        prediction = predict(input_data)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
