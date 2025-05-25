from pydantic import BaseModel, Field
from typing import List, Optional, Union


class PatientData(BaseModel):
    """Modelo para dados de um paciente"""
    age: float = Field(..., description="Idade do paciente")
    sex: Union[str, int] = Field(..., description="Sexo (Male/Female ou 1/0)")
    height: float = Field(..., description="Altura em cm")
    weight: float = Field(..., description="Peso em kg")
    waistline: float = Field(..., description="Circunferência da cintura")
    sight_left: float = Field(..., description="Acuidade visual olho esquerdo")
    sight_right: float = Field(..., description="Acuidade visual olho direito")
    hear_left: int = Field(..., description="Audição ouvido esquerdo (1=normal, 2=anormal)")
    hear_right: int = Field(..., description="Audição ouvido direito (1=normal, 2=anormal)")
    SBP: float = Field(..., description="Pressão sistólica (mmHg)")
    DBP: float = Field(..., description="Pressão diastólica (mmHg)")
    BLDS: float = Field(..., description="Glicemia em jejum (mg/dL)")
    tot_chole: float = Field(..., description="Colesterol total (mg/dL)")
    HDL_chole: float = Field(..., description="Colesterol HDL (mg/dL)")
    LDL_chole: float = Field(..., description="Colesterol LDL (mg/dL)")
    triglyceride: float = Field(..., description="Triglicerídeos (mg/dL)")
    hemoglobin: float = Field(..., description="Hemoglobina (g/dL)")
    urine_protein: int = Field(..., description="Proteína na urina (1-6)")
    serum_creatinine: float = Field(..., description="Creatinina sérica (mg/dL)")
    SGOT_AST: float = Field(..., description="SGOT/AST (UI/L)")
    SGOT_ALT: float = Field(..., description="SGOT/ALT (UI/L)")
    gamma_GTP: float = Field(..., description="Gamma-GTP (UI/L)")
    SMK_stat_type_cd: int = Field(..., description="Status de fumante (1=nunca, 2=ex-fumante, 3=fumante)")

    class Config:
        json_schema_extra = {
            "example": {
                "sex": "Male",
                "age": 35,
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
        }


class BatchPatientData(BaseModel):
    """Modelo para múltiplos pacientes"""
    patients: List[PatientData] = Field(..., description="Lista de dados de pacientes")


class PredictionResponse(BaseModel):
    """Modelo para resposta de predição"""
    prediction: int = Field(..., description="Predição (0=não bebe, 1=bebe)")
    prediction_label: str = Field(..., description="Label da predição")
    probability: Optional[float] = Field(None, description="Probabilidade da classe positiva")


class BatchPredictionResponse(BaseModel):
    """Modelo para resposta de predição em lote"""
    predictions: List[PredictionResponse] = Field(..., description="Lista de predições")


class HealthResponse(BaseModel):
    """Modelo para resposta de health check"""
    status: str = Field(..., description="Status do serviço")
    model_loaded: bool = Field(..., description="Se o modelo está carregado")


class ModelInfoResponse(BaseModel):
    """Modelo para informações do modelo"""
    model_loaded: bool = Field(..., description="Se o modelo está carregado")
    features_trained: bool = Field(..., description="Se as features estão treinadas")
    model_type: str = Field(..., description="Tipo do modelo")


class CleanupResponse(BaseModel):
    """Modelo para resposta da operação de limpeza"""
    status: str = Field(..., description="Status da operação")
    message: str = Field(..., description="Mensagem detalhada")
    cleaned_items: List[str] = Field(..., description="Lista de itens limpos")
    errors: Optional[List[str]] = Field(default=[], description="Lista de erros encontrados") 