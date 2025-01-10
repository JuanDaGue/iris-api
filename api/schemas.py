from pydantic import BaseModel


class PredictInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictOutput(BaseModel):
    prediction: str
