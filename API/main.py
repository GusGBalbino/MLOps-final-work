from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from routers import predictions, health, admin
from services.prediction_service import prediction_service

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação"""
    # Startup
    logger.info("Iniciando aplicação FastAPI...")
    try:
        # Carregar artefatos na inicialização
        prediction_service.load_artifacts()
        logger.info("Artefatos carregados com sucesso")
    except Exception as e:
        logger.warning(f"Não foi possível carregar artefatos: {e}")
    
    yield
    
    # Shutdown
    logger.info("Encerrando aplicação FastAPI...")


# Criar aplicação FastAPI
app = FastAPI(
    title="API de Predição de Consumo de Álcool",
    description="API para predição de consumo de álcool baseada em dados de saúde usando Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Incluir routers
app.include_router(health.router, tags=["Health"])
app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 