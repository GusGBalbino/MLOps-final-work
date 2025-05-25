from fastapi import APIRouter, HTTPException, status
from schemas import CleanupResponse
from services.prediction_service import prediction_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.delete("/cleanup", response_model=CleanupResponse)
async def cleanup_all_models():
    """
    Limpa todos os modelos e artefatos para começar do zero
    
    Esta operação irá:
    - Resetar o estado do serviço de predição
    - Remover todos os artefatos de modelos salvos
    - Remover todos os artefatos de preprocessamento
    - Limpar os experimentos e runs do MLFlow
    - Remover modelos registrados no MLFlow
    
    **Atenção**: Esta operação é irreversível!
    """
    try:
        logger.info("Iniciando limpeza completa de modelos e artefatos")
        result = prediction_service.cleanup_all_artifacts()
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["message"]
            )
        
        logger.info(f"Limpeza concluída: {result['message']}")
        return CleanupResponse(**result)
        
    except Exception as e:
        logger.error(f"Erro na operação de limpeza: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro na operação de limpeza: {str(e)}"
        )


@router.post("/reset-service")
async def reset_service():
    """
    Reseta apenas o estado do serviço sem deletar arquivos
    """
    try:
        logger.info("Resetando estado do serviço")
        
        # Resetar apenas o estado em memória
        prediction_service.model = None
        prediction_service.model_loaded = False
        prediction_service.features_trained = False
        prediction_service.preprocessor = prediction_service.preprocessor.__class__()
        
        return {
            "status": "success",
            "message": "Estado do serviço resetado com sucesso",
            "model_loaded": prediction_service.model_loaded,
            "features_trained": prediction_service.features_trained
        }
        
    except Exception as e:
        logger.error(f"Erro ao resetar serviço: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao resetar serviço: {str(e)}"
        ) 