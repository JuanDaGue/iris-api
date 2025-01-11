"""
Módulo: Entrenador de Modelo de Árbol de Decisión para el Conjunto de Datos Iris

Este módulo realiza las siguientes tareas:
1. Carga el conjunto de datos Iris.
2. Divide el conjunto de datos en conjuntos de entrenamiento y prueba.
3. Entrena un Clasificador de Árbol de Decisión con los datos de entrenamiento.
4. Guarda el modelo entrenado como un archivo serializado para su uso futuro.

Dependencias:
- sklearn
- pickle
- os

Uso:
Ejecute el script para entrenar el modelo y guardarlo
en el directorio especificado.
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle
import os

# Cargar el conjunto de datos Iris


def load_data():
    """
    Carga el conjunto de datos Iris.

    Retorna:
        tuple: Una tupla que contiene las características (X) y las etiquetas objetivo (y).
    """
    data = datasets.load_iris()
    X = data.data
    y = data.target
    return X, y


# Dividir el conjunto de datos
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Divide el conjunto de datos en conjuntos de entrenamiento y prueba.

    Argumentos:
        X (array-like): Características del conjunto de datos.
        y (array-like): Etiquetas objetivo del conjunto de datos.
        test_size (float): Proporción del conjunto de datos que se
        incluirá en la división de prueba.
        random_state (int): Semilla aleatoria para garantizar reproducibilidad.

    Retorna:
        tuple: Divisiones de entrenamiento y prueba
        (X_train, X_test, y_train, y_test).
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Entrenar el modelo
def train_model(X_train, y_train, random_state=42):
    """
    Entrena un Clasificador de Árbol de Decisión con los
    datos de entrenamiento.

    Argumentos:
        X_train (array-like): Características de entrenamiento.
        y_train (array-like): Etiquetas objetivo de entrenamiento.
        random_state (int): Semilla aleatoria para garantizar reproducibilidad.

    Retorna:
        DecisionTreeClassifier: El modelo de Árbol de Decisión entrenado.
    """
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


# Guardar el modelo
def save_model(model, directory="models/data/processed/", filename="model.pkl"):
    """
    Guarda el modelo entrenado en un directorio especificado.

    Argumentos:
        model (DecisionTreeClassifier): El modelo entrenado a guardar.
        directory (str): Directorio donde se guardará el modelo.
        filename (str): Nombre del archivo para guardar el modelo.

    Retorna:
        str: Ruta completa al archivo del modelo guardado.
    """
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
    return file_path


if __name__ == "__main__":
    # Paso 1: Cargar datos
    X, y = load_data()

    # Paso 2: Dividir datos
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Paso 3: Entrenar el modelo
    model = train_model(X_train, y_train)

    # Paso 4: Guardar el modelo
    model_path = save_model(model)
    print(f"Modelo guardado en {model_path}")
