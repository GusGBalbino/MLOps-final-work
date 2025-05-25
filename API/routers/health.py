from fastapi import APIRouter
from schemas import HealthResponse, ModelInfoResponse
from services.prediction_service import prediction_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de health check"""
    return HealthResponse(
        status="healthy",
        model_loaded=prediction_service.model_loaded
    )


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Obtém informações sobre o modelo carregado"""
    model_info = prediction_service.get_model_info()
    return ModelInfoResponse(**model_info) 