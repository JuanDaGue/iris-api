from fastapi import APIRouter, HTTPException
from api.schemas import PredictInput, PredictOutput
from api.services import predict

router = APIRouter()


@router.post("/predict", response_model=PredictOutput)
async def predict_endpoint(input_data: PredictInput):
    """
    Endpoint to predict the class of Iris flowers based on input features.
    """
    try:
        prediction = predict(input_data)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
