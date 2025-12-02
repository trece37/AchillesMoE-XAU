#!/usr/bin/env python3
"""
Predictor custom para Vertex AI - AchillesMoE Bot
Endpoint FastAPI que carga el modelo MoE y el scaler para hacer predicciones de XAUUSD.
"""

import os
import logging
from typing import List, Dict, Any
import numpy as np
import tensorflow as tf
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Achilles MoE Predictor",
    description="Endpoint de predicción para trading XAUUSD con arquitectura MoE",
    version="1.0.0"
)

# Variables globales para modelo y scaler
model = None
scaler = None

# Configuración
MODEL_DIR = os.getenv("AIP_STORAGE_URI", "./achilles_moe_export")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


class PredictionRequest(BaseModel):
    """Schema para request de predicción"""
    instances: List[List[float]]  # Array 2D: [[feature1, feature2, ...], ...]


class PredictionResponse(BaseModel):
    """Schema para response de predicción"""
    predictions: List[float]
    model_version: str = "1.0.0"


@app.on_event("startup")
async def load_model_and_scaler():
    """
    Carga el modelo TensorFlow y el scaler al iniciar el servidor.
    Se ejecuta una sola vez cuando el contenedor arranca.
    """
    global model, scaler
    
    try:
        logger.info(f"Cargando modelo desde: {MODEL_DIR}")
        model = tf.keras.models.load_model(MODEL_DIR)
        logger.info("✅ Modelo MoE cargado exitosamente")
        
        logger.info(f"Cargando scaler desde: {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)
        logger.info("✅ Scaler cargado exitosamente")
        
        # Validar que el modelo esté listo
        dummy_input = np.zeros((1, 60, 9))  # (batch, timesteps, features)
        _ = model.predict(dummy_input, verbose=0)
        logger.info("✅ Modelo validado con predicción dummy")
        
    except Exception as e:
        logger.error(f"❌ Error al cargar modelo o scaler: {e}")
        raise RuntimeError(f"No se pudo inicializar el predictor: {e}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint para Kubernetes/Vertex AI.
    Verifica que el modelo y scaler estén cargados.
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo o scaler no cargados"
        )
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Endpoint principal de predicción.
    
    Input esperado:
    {
      "instances": [
        [feature1, feature2, ..., feature9],  # 60 timesteps
        ...
      ]
    }
    
    Output:
    {
      "predictions": [precio_predicho],
      "model_version": "1.0.0"
    }
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Intenta más tarde."
        )
    
    try:
        # Convertir a numpy array
        features = np.array(request.instances)  # Shape: (n_samples, n_features)
        
        logger.info(f"Recibido request con shape: {features.shape}")
        
        # Validar dimensiones
        if features.ndim != 2:
            raise ValueError(f"Se esperaba array 2D, recibido shape: {features.shape}")
        
        # Normalizar con el scaler
        features_scaled = scaler.transform(features)
        logger.info(f"Features normalizadas: {features_scaled.shape}")
        
        # Reshape para LSTM: (n_samples, timesteps, features)
        # Si recibimos 60 muestras con 9 features cada una:
        # (60, 9) -> (1, 60, 9) para el modelo
        if features_scaled.shape[0] == 60:  # Timesteps esperados
            features_reshaped = features_scaled.reshape(1, 60, -1)
        else:
            # Si es una sola muestra con múltiples timesteps ya
            features_reshaped = features_scaled.reshape(1, -1, features_scaled.shape[-1])
        
        logger.info(f"Shape final para modelo: {features_reshaped.shape}")
        
        # Hacer predicción
        predictions = model.predict(features_reshaped, verbose=0)
        
        # Convertir a lista de floats
        predictions_list = predictions.flatten().tolist()
        
        logger.info(f"✅ Predicción exitosa: {predictions_list}")
        
        return PredictionResponse(
            predictions=predictions_list,
            model_version="1.0.0"
        )
        
    except ValueError as ve:
        logger.error(f"Error de validación: {ve}")
        raise HTTPException(
            status_code=400,
            detail=f"Datos inválidos: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Error en predicción: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error interno: {str(e)}"
        )


@app.get("/")
async def root():
    """
    Endpoint raíz con información del servicio.
    """
    return {
        "service": "Achilles MoE Predictor",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Ejecutar servidor (para testing local)
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
