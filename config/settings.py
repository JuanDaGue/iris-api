import os


class Settings:
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/data/processed/model.pkl")


settings = Settings()
