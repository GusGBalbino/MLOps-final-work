# API de Predição de Consumo de Álcool 🍺

Uma API moderna construída com **FastAPI** para predição de consumo de álcool baseada em dados de saúde, utilizando **Machine Learning** e o padrão de design **Abstract Factory**.

## 📁 Estrutura do Projeto

```
mlops-final/
├── API/
│   ├── Features/
│   │   ├── Artefatos/          # Arquivos de preprocessamento (scaler.pkl, ordinal.pkl)
│   │   ├── __init__.py
│   │   └── preprocessamento.py # Módulo de preprocessamento completo
│   ├── Modelo/
│   │   ├── Artefatos/          # Modelos treinados (modelo.bin)
│   │   ├── __init__.py
│   │   └── modelo.py           # Implementação Abstract Factory
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── health.py           # Endpoints de health check
│   │   └── predictions.py      # Endpoints de predição
│   ├── services/
│   │   ├── __init__.py
│   │   └── prediction_service.py # Lógica de negócio
│   ├── __init__.py
│   ├── main.py                 # Aplicação FastAPI principal
│   ├── schemas.py              # Modelos Pydantic
│   └── train.py                # Script de treinamento
├── exemplo_uso_fastapi.py      # Exemplo de uso da API
├── requirements.txt            # Dependências do projeto
├── smoking_drinking.parquet    # Dataset
└── README.md                   # Este arquivo
```

## 🏗️ Arquitetura

### Padrão Abstract Factory
O projeto implementa o padrão Abstract Factory para criação de modelos de ML:

- **ModelFactory**: Factory abstrata
- **RandomForestFactory**: Factory para Random Forest
- **LogisticRegressionFactory**: Factory para Regressão Logística
- **SVMFactory**: Factory para SVM

### Componentes Principais

1. **main.py**: Aplicação FastAPI principal com configuração de routers
2. **schemas.py**: Modelos Pydantic para validação de dados
3. **routers/**: Endpoints organizados por funcionalidade
4. **services/**: Lógica de negócio e processamento
5. **Features/**: Preprocessamento de dados
6. **Modelo/**: Modelos de ML com Abstract Factory

## 📖 Documentação da API

### Swagger UI (Interativa)
- **URL**: `http://localhost:8000/docs`
- Interface interativa para testar endpoints

### MLFlow
- **URL**: `http://localhost:5000/`
- App MLFlow com modelos

## 🔗 Endpoints

### 1. Health Check
```
GET /health
```
Verifica se a API está funcionando e se o modelo está carregado.

### 2. Informações do Modelo
```
GET /model/info
```
Retorna informações sobre o modelo carregado.

### 3. Predição Única
```
POST /api/v1/predict
```
Faz predição para um único paciente.

### 4. Limpeza de todos os modelos
```
DELETE /api/v1/admin/cleanup
```
Limpa todos os modelos e artefatos para começar do zero.

### 5. Reset de serviços
```
POST /api/v1/admin/reset-service
```
Reseta apenas o estado do serviço sem deletar arquivos


**Exemplo de payload:**
```json
{
  "age": 35,
  "sex": "Male",
  "height": 175,
  "weight": 70,
  "waistline": 85,
  "sight_left": 1.0,
  "sight_right": 1.0,
  "hear_left": 1,
  "hear_right": 1,
  "SBP": 120,
  "DBP": 80,
  "BLDS": 90,
  "tot_chole": 180,
  "HDL_chole": 50,
  "LDL_chole": 110,
  "triglyceride": 100,
  "hemoglobin": 14.5,
  "urine_protein": 1,
  "serum_creatinine": 1.0,
  "SGOT_AST": 25,
  "SGOT_ALT": 20,
  "gamma_GTP": 30,
  "SMK_stat_type_cd": 1
}
```


## 📊 Variáveis do Dataset

| Variável | Descrição |
|----------|-----------|
| `age` | Idade do paciente |
| `sex` | Sexo (Male/Female ou 1/0) |
| `height` | Altura em cm |
| `weight` | Peso em kg |
| `waistline` | Circunferência da cintura |
| `sight_left/right` | Acuidade visual |
| `hear_left/right` | Audição (1=normal, 2=anormal) |
| `SBP/DBP` | Pressão sistólica/diastólica (mmHg) |
| `BLDS` | Glicemia em jejum (mg/dL) |
| `tot_chole` | Colesterol total (mg/dL) |
| `HDL_chole` | Colesterol HDL (mg/dL) |
| `LDL_chole` | Colesterol LDL (mg/dL) |
| `triglyceride` | Triglicerídeos (mg/dL) |
| `hemoglobin` | Hemoglobina (g/dL) |
| `urine_protein` | Proteína na urina (1-6) |
| `serum_creatinine` | Creatinina sérica (mg/dL) |
| `SGOT_AST` | SGOT/AST (UI/L) |
| `SGOT_ALT` | SGOT/ALT (UI/L) |
| `gamma_GTP` | Gamma-GTP (UI/L) |
| `SMK_stat_type_cd` | Status fumante (1=nunca, 2=ex, 3=atual) |

## 🎯 Features Criadas Automaticamente

O preprocessador cria automaticamente:

1. **IMC**: Índice de Massa Corporal
2. **organ_overload**: Indicador de sobrecarga de órgãos (SGOT_AST > 40)
3. **fail_hepatic_func**: Indicador de falha na função hepática baseado em sexo e gamma_GTP

## 📈 MLflow Integration

O projeto integra com MLflow para:
- Tracking de experimentos
- Logging de métricas
- Registro de modelos
- Comparação de resultados

## 🏛️ Arquitetura FastAPI

A API segue as melhores práticas do FastAPI:

- **Separation of Concerns**: Separação clara entre routers, services e schemas
- **Dependency Injection**: Injeção de dependências para services
- **Pydantic Validation**: Validação automática de dados
- **Async Support**: Suporte para operações assíncronas
- **Auto Documentation**: Documentação automática com Swagger UI

## 🔒 Validação de Dados

A API usa **Pydantic** para:
- Validação automática de tipos
- Documentação de campos
- Exemplos na documentação
- Mensagens de erro claras


## 🐳 Execução com Docker (Recomendado)

### Pré-requisitos
- Docker
- Docker Compose

### 🚀 Início Rápido com Docker

```bash
# 1. Construir as imagens
./docker-scripts.sh build

# 2. Treinar os modelos
./docker-scripts.sh train

# 3. Iniciar API + MLflow
./docker-scripts.sh start
```

**Pronto!** Sua aplicação está rodando:
- 📖 **API**: http://localhost:8000
- 📊 **MLflow**: http://localhost:5000

### 🛠️ Comandos Docker Disponíveis

```bash
#Menu
./docker-scripts.sh

# Inicialização
./docker-scripts.sh build      # Constrói as imagens
./docker-scripts.sh start      # Inicia API + MLflow
./docker-scripts.sh restart    # Reinicia serviços

# Treinamento
./docker-scripts.sh train      # Treina modelos
./docker-scripts.sh retrain    # Re-treina (limpa artefatos)

# Monitoramento
./docker-scripts.sh status     # Status dos containers
./docker-scripts.sh logs-api   # Logs da API
./docker-scripts.sh logs-mlflow # Logs do MLflow

# Testes
./docker-scripts.sh test       # Executa testes da API

# Utilitários
./docker-scripts.sh shell-api  # Shell no container da API
./docker-scripts.sh stop       # Para serviços
./docker-scripts.sh clean      # Limpa tudo
```

### 🔄 Fluxo de Desenvolvimento com Docker

```bash
# Desenvolvimento iterativo
./docker-scripts.sh retrain    # Re-treina modelos
./docker-scripts.sh restart    # Reinicia API
./docker-scripts.sh test       # Testa API

# Monitoramento
./docker-scripts.sh logs-api   # Acompanha logs
```

## 🏗️ Arquitetura Docker

### Serviços

1. **API (api)**: Aplicação FastAPI principal
   - Porta: 8000
   - Volumes: Artefatos de modelos e features

2. **MLflow (mlflow)**: Interface de tracking
   - Porta: 5000  
   - Volumes: Dados de experimentos

3. **Trainer (trainer)**: Treinamento de modelos
   - Execução sob demanda
   - Volumes: Todos os artefatos e dados

### Volumes Persistentes

```yaml
volumes:
  - ./API/Features/Artefatos   # Artefatos de preprocessamento
  - ./API/Modelo/Artefatos     # Modelos treinados
  - ./mlruns                   # Dados do MLflow
  - ./smoking_drinking.parquet # Dataset
```

### Rede

Todos os serviços rodam na mesma rede Docker (`mlops-network`) permitindo comunicação entre containers.

## 📊 MLflow Integration

### Acesso ao MLflow
- **URL**: http://localhost:5000
- **Experimentos**: Visualize comparações entre modelos
- **Métricas**: Accuracy, Precision, Recall, F1-score
- **Artefatos**: Modelos, parâmetros, dados