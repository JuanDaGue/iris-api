"""
Módulo: Configuración de la API para Clasificación de Iris

Este módulo define la aplicación FastAPI que proporciona una API REST
para predecir la clase de flores Iris basándose en un modelo preentrenado.

Características:
- Documentación automática accesible en la raíz del servidor.
- Integración de rutas definidas en `api.routes`.

Dependencias:
- fastapi
- api.routes: Contiene las rutas para la funcionalidad de predicción.

Uso:
Ejecute este módulo para iniciar la API.
"""

from fastapi import FastAPI
from api.routes import router

# Crear instancia de FastAPI
app = FastAPI(
    title="Iris Classification API",
    description="API para predecir la clase de flores Iris usando un modelo entrenado.",
    version="1.0.0",
    docs_url="/",  # Cambia la ruta de la documentación a la raíz
)

# Incluir rutas del módulo de rutas
app.include_router(router)
