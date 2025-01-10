from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Iris Classification API",
    description="API to predict the class of Iris flowers using a trained model.",
    version="1.0.0",
)

app.include_router(router)
