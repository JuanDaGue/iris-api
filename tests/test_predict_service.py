import os
import pytest
from unittest import mock
from api.schemas import PredictInput
from api.services import predict  # Asegúrate de que esta sea la ruta correcta


# Mock de la carga del modelo para evitar la carga real del archivo pickle
@pytest.fixture
def mock_model_load():
    with mock.patch("api.services.pickle.load") as mock_load:
        mock_model = mock.MagicMock()
        mock_load.return_value = mock_model
        yield mock_model


# Mock de la variable de entorno
@pytest.fixture
def mock_env_variable():
    with mock.patch.dict(os.environ, {"MODEL_PATH": "models/data/processed/model.pkl"}):
        yield


# Prueba de la función predict con datos de entrada válidos
def test_predict_valid_input(mock_model_load, mock_env_variable):
    # Configura el mock para que el modelo prediga un valor específico
    mock_model_load.predict.return_value = [1]  # Predicción: 'versicolor'
    print('MockModel', mock_model_load)
    # Datos de entrada válidos
    input_data = PredictInput(
        sepal_length=5.1,
        sepal_width=3.5,
        petal_length=1.4,
        petal_width=0.2
    )

    # Llama a la función de predicción
    species_name = predict(input_data)

    # Verifica que la predicción devuelva el nombre correcto
    assert species_name == "setosa"


# Prueba de la función predict con un valor de predicción desconocido
def test_predict_unknown_species(mock_model_load, mock_env_variable):
    # Configura el mock para que el modelo prediga un valor desconocido
    mock_model_load.predict.return_value = [99]  # Predicción desconocida

    # Datos de entrada válidos
    input_data = PredictInput(
        sepal_length=5.1,
        sepal_width=3.5,
        petal_length=1.4,
        petal_width=0.2
    )

    # Llama a la función de predicción
    species_name = predict(input_data)

    # Verifica que se devuelva "Unknown" si la predicción es desconocida
    assert species_name == "Unknown"


# Prueba de la función predict con un valor de entrada incorrecto
def test_predict_invalid_input(mock_model_load, mock_env_variable):
    # Configura el mock para que el modelo prediga un valor válido
    mock_model_load.predict.return_value = [0]  # Predicción: 'setosa'

    # Datos de entrada inválidos (ejemplo de un tipo de dato incorrecto)
    input_data = PredictInput(
        sepal_length="invalid",  # Tipo de dato incorrecto
        sepal_width=3.5,
        petal_length=1.4,
        petal_width=0.2
    )

    # Verifica que se genere un error en la predicción
    # debido a los datos inválidos
    with pytest.raises(ValueError):
        predict(input_data)
