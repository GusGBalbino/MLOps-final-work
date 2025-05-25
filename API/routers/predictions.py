from fastapi import APIRouter
from schemas import (
    PatientData, 
    BatchPatientData, 
    PredictionResponse, 
    BatchPredictionResponse
)
from services.prediction_service import prediction_service

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(patient_data: PatientData):
    """
    Faz predição para um único paciente
    
    - **patient_data**: Dados completos do paciente incluindo exames
    - **returns**: Predição se o paciente bebe ou não, com probabilidade
    """
    return prediction_service.predict_single(patient_data)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_data: BatchPatientData):
    """
    Faz predições para múltiplos pacientes em lote
    
    - **batch_data**: Lista de dados de pacientes
    - **returns**: Lista de predições para cada paciente
    """
    predictions = prediction_service.predict_batch(batch_data.patients)
    return BatchPredictionResponse(predictions=predictions) 