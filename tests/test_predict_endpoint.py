"""
Módulo: Pruebas para la API de Clasificación de Iris

Este módulo contiene pruebas para verificar el funcionamiento de la API de clasificación de flores Iris.
Se utilizan datos válidos e inválidos, así como pruebas para simular errores del servidor.

Características:
- Pruebas unitarias para verificar el endpoint `/predict`.
- Fixtures para gestionar datos de entrada válidos e inválidos.
- Uso de `pytest` y `fastapi.testclient` para pruebas.

Dependencias:
- pytest
- fastapi.testclient
- mocker (para simular errores en servicios externos).

Uso:
Ejecute las pruebas utilizando `pytest`.
"""

import pytest
from fastapi.testclient import TestClient
from main import app  # Importa la instancia de la aplicación FastAPI
from api.schemas import PredictInput

# Crear cliente de prueba para la aplicación
client = TestClient(app)


@pytest.fixture
def valid_input_data():
    """Fixture para datos de entrada válidos."""
    return {
        "sepal_length": 5.8,
        "sepal_width": 2.7,
        "petal_length": 3.9,
        "petal_width": 1.2
    }


@pytest.fixture
def invalid_input_data():
    """Fixture para datos de entrada inválidos."""
    return {
        "feature1": "invalid",
        "feature2": 3.5,
        "feature3": 1.4,
        # Faltan campos
    }


def test_predict_endpoint_valid_input(valid_input_data):
    """Prueba el endpoint con datos de entrada válidos."""
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_endpoint_invalid_input(invalid_input_data):
    """Prueba el endpoint con datos de entrada inválidos."""
    response = client.post("/predict", json=invalid_input_data)
    assert response.status_code == 422  # Error de validación de FastAPI


def test_predict_endpoint_server_error(mocker, valid_input_data):
    """Simula un error del servidor durante la predicción."""
    mocker.patch("api.services.predict", side_effect=Exception("Simulated error"))

    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert response.json()["detail"] == "Error en la predicción: Simulated error"
