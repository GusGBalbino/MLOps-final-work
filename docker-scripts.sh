#!/bin/bash

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐳 MLOps Docker Management Scripts${NC}"
echo "======================================"

# Detectar comando do Docker Compose
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
elif docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    echo -e "${RED}❌ Docker Compose não encontrado!${NC}"
    echo "Instale Docker Compose ou Docker Desktop"
    exit 1
fi

function show_help() {
    echo -e "${YELLOW}Comandos disponíveis:${NC}"
    echo ""
    echo -e "${GREEN}🚀 Inicialização:${NC}"
    echo "  build     - Constrói as imagens Docker"
    echo "  start     - Inicia API + MLflow"
    echo "  restart   - Reinicia todos os serviços"
    echo ""
    echo -e "${GREEN}🤖 Treinamento:${NC}"
    echo "  train     - Treina os modelos"
    echo "  retrain   - Re-treina os modelos (limpa artefatos)"
    echo ""
    echo -e "${GREEN}📊 Monitoramento:${NC}"
    echo "  logs-api  - Mostra logs da API"
    echo "  logs-mlflow - Mostra logs do MLflow"
    echo "  status    - Status dos containers"
    echo ""
    echo -e "${GREEN}🧪 Testes:${NC}"
    echo "  test      - Executa testes da API"
    echo ""
    echo -e "${GREEN}🛑 Parada:${NC}"
    echo "  stop      - Para todos os serviços"
    echo "  clean     - Para e remove containers/volumes"
    echo ""
    echo -e "${GREEN}🔧 Utilitários:${NC}"
    echo "  shell-api - Abre shell no container da API"
    echo "  shell-trainer - Abre shell no container de treinamento"
}

case "$1" in
    "build")
        echo -e "${BLUE}🔨 Construindo imagens Docker...${NC}"
        $DOCKER_COMPOSE build
        ;;
    "start")
        echo -e "${BLUE}🚀 Iniciando API + MLflow...${NC}"
        $DOCKER_COMPOSE up -d api mlflow
        echo -e "${GREEN}✅ Serviços iniciados!${NC}"
        echo -e "${YELLOW}📖 API: http://localhost:8000${NC}"
        echo -e "${YELLOW}📊 MLflow: http://localhost:5000${NC}"
        ;;
    "restart")
        echo -e "${BLUE}🔄 Reiniciando serviços...${NC}"
        $DOCKER_COMPOSE restart api mlflow
        ;;
    "train")
        echo -e "${BLUE}🤖 Iniciando treinamento dos modelos...${NC}"
        $DOCKER_COMPOSE --profile training run --rm trainer python train.py --data ../smoking_drinking.parquet
        echo -e "${GREEN}✅ Treinamento concluído!${NC}"
        ;;
    "retrain")
        echo -e "${BLUE}🗑️ Limpando artefatos antigos...${NC}"
        sudo rm -rf API/Features/Artefatos/* API/Modelo/Artefatos/* 2>/dev/null || true
        echo -e "${BLUE}🤖 Re-treinando modelos...${NC}"
        $DOCKER_COMPOSE --profile training run --rm trainer python train.py --data ../smoking_drinking.parquet
        echo -e "${GREEN}✅ Re-treinamento concluído!${NC}"
        ;;
    "logs-api")
        echo -e "${BLUE}📋 Logs da API:${NC}"
        $DOCKER_COMPOSE logs -f api
        ;;
    "logs-mlflow")
        echo -e "${BLUE}📋 Logs do MLflow:${NC}"
        $DOCKER_COMPOSE logs -f mlflow
        ;;
    "status")
        echo -e "${BLUE}📊 Status dos containers:${NC}"
        $DOCKER_COMPOSE ps
        ;;
    "test")
        echo -e "${BLUE}🧪 Executando testes da API...${NC}"
        sleep 3  # Aguarda API inicializar
        $DOCKER_COMPOSE exec api python ../exemplo_uso_fastapi.py
        ;;
    "stop")
        echo -e "${BLUE}🛑 Parando serviços...${NC}"
        $DOCKER_COMPOSE down
        echo -e "${GREEN}✅ Serviços parados!${NC}"
        ;;
    "clean")
        echo -e "${BLUE}🧹 Limpando containers e volumes do projeto...${NC}"
        $DOCKER_COMPOSE down -v --remove-orphans
        
        # Remove apenas as imagens do projeto
        echo -e "${BLUE}🗑️ Removendo imagens do projeto...${NC}"
        docker rmi mlops-final-api mlops-final-mlflow mlops-final-trainer 2>/dev/null || true
        docker rmi $(docker images -q -f "label=project=mlops-final") 2>/dev/null || true
        
        # Remove network específica do projeto
        docker network rm mlops-final_mlops-network 2>/dev/null || true
        
        echo -e "${GREEN}✅ Limpeza do projeto concluída!${NC}"
        ;;
    "shell-api")
        echo -e "${BLUE}🐚 Abrindo shell no container da API...${NC}"
        $DOCKER_COMPOSE exec api bash
        ;;
    "shell-trainer")
        echo -e "${BLUE}🐚 Abrindo shell no container de treinamento...${NC}"
        $DOCKER_COMPOSE --profile training run --rm trainer bash
        ;;
    *)
        show_help
        ;;
esac 