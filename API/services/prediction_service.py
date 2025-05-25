import pandas as pd
import joblib
import logging
import os
import shutil
from pathlib import Path
from typing import List
from fastapi import HTTPException, status

from Features.preprocessamento import DataPreprocessor
from schemas import PatientData, PredictionResponse

logger = logging.getLogger(__name__)


class PredictionService:
    """Serviço responsável pelas predições de consumo de álcool"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.model_loaded = False
        self.features_trained = False
    
    def load_artifacts(self, 
                      model_path: str = "API/Modelo/Artefatos/modelo.bin",
                      preprocessor_path: str = "API/Features/Artefatos"):
        """Carrega os artefatos do modelo e preprocessador"""
        try:
            # Carregar preprocessador
            self.preprocessor.load_artifacts(preprocessor_path)
            self.features_trained = True
            logger.info("Artefatos de preprocessamento carregados")
            
            # Carregar modelo
            self.model = joblib.load(model_path)
            self.model_loaded = True
            logger.info("Modelo carregado")
            
        except Exception as e:
            logger.error(f"Erro ao carregar artefatos: {str(e)}")
            raise
    
    def cleanup_all_artifacts(self) -> dict:
        """Remove todos os modelos e artefatos para começar do zero"""
        cleaned_items = []
        errors = []
        
        try:
            # Resetar estado do serviço
            self.model = None
            self.model_loaded = False
            self.features_trained = False
            self.preprocessor = DataPreprocessor()
            cleaned_items.append("Estado do serviço resetado")
            
            # Diretórios a serem limpos
            cleanup_paths = [
                "API/Modelo/Artefatos",
                "API/Features/Artefatos", 
                "API/mlruns",
                "mlruns"  # MLRuns na raiz também
            ]
            
            for path in cleanup_paths:
                if os.path.exists(path):
                    try:
                        if os.path.isdir(path):
                            # Remove todo o conteúdo mas mantém o diretório
                            for item in os.listdir(path):
                                item_path = os.path.join(path, item)
                                if os.path.isdir(item_path):
                                    shutil.rmtree(item_path)
                                else:
                                    os.remove(item_path)
                            cleaned_items.append(f"Conteúdo do diretório {path} removido")
                        else:
                            os.remove(path)
                            cleaned_items.append(f"Arquivo {path} removido")
                    except Exception as e:
                        error_msg = f"Erro ao limpar {path}: {str(e)}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                else:
                    cleaned_items.append(f"Diretório {path} não existe (já limpo)")
            
            # Limpar modelos registrados no MLFlow (se existir)
            try:
                import mlflow
                mlflow.set_tracking_uri("file:./API/mlruns")
                
                # Listar todos os experimentos
                experiments = mlflow.search_experiments()
                for exp in experiments:
                    if exp.name != "Default":  # Não deletar experimento padrão
                        try:
                            mlflow.delete_experiment(exp.experiment_id)
                            cleaned_items.append(f"Experimento MLFlow {exp.name} deletado")
                        except Exception as e:
                            errors.append(f"Erro ao deletar experimento {exp.name}: {str(e)}")
                            
                # Limpar modelos registrados
                try:
                    from mlflow.tracking import MlflowClient
                    client = MlflowClient()
                    registered_models = client.search_registered_models()
                    for model in registered_models:
                        try:
                            client.delete_registered_model(model.name)
                            cleaned_items.append(f"Modelo registrado {model.name} deletado")
                        except Exception as e:
                            errors.append(f"Erro ao deletar modelo {model.name}: {str(e)}")
                except Exception as e:
                    errors.append(f"Erro ao acessar modelos registrados: {str(e)}")
                    
            except ImportError:
                cleaned_items.append("MLFlow não disponível - limpeza manual dos diretórios realizada")
            except Exception as e:
                errors.append(f"Erro ao limpar artefatos do MLFlow: {str(e)}")
            
            status_msg = "success" if not errors else "partial_success"
            message = "Limpeza concluída com sucesso" if not errors else f"Limpeza concluída com {len(errors)} erro(s)"
            
            return {
                "status": status_msg,
                "message": message,
                "cleaned_items": cleaned_items,
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Erro crítico na limpeza: {str(e)}")
            return {
                "status": "error",
                "message": f"Erro crítico na limpeza: {str(e)}",
                "cleaned_items": cleaned_items,
                "errors": errors + [str(e)]
            }
    
    def _validate_service_ready(self):
        """Valida se o serviço está pronto para fazer predições"""
        if not self.model_loaded or not self.features_trained:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modelo e preprocessador devem ser carregados primeiro"
            )
    
    def _create_ordered_dataframe(self, data_dict: dict) -> pd.DataFrame:
        """Cria DataFrame com as colunas na ordem original do dataset"""
        # Ordem original conforme dataset smoking_drinking.parquet
        original_order = [
            'sex', 'age', 'height', 'weight', 'waistline', 'sight_left', 'sight_right',
            'hear_left', 'hear_right', 'SBP', 'DBP', 'BLDS', 'tot_chole', 'HDL_chole',
            'LDL_chole', 'triglyceride', 'hemoglobin', 'urine_protein', 'serum_creatinine',
            'SGOT_AST', 'SGOT_ALT', 'gamma_GTP', 'SMK_stat_type_cd'
        ]
        
        # Criar DataFrame com colunas na ordem correta
        ordered_data = {}
        for col in original_order:
            if col in data_dict:
                ordered_data[col] = data_dict[col]
        
        return pd.DataFrame([ordered_data])

    def predict_single(self, patient_data: PatientData) -> PredictionResponse:
        """Faz predição para um único paciente"""
        self._validate_service_ready()
        
        try:
            # Converter para dicionário e criar DataFrame com ordem correta
            data_dict = patient_data.model_dump()
            df = self._create_ordered_dataframe(data_dict)
            
            # Preprocessar dados
            df_processed = self.preprocessor.process_data(df, is_training=False)
            
            # Fazer predição
            prediction = self.model.predict(df_processed)[0]
            
            # Obter probabilidades se disponível
            try:
                probabilities = self.model.predict_proba(df_processed)[0]
                probability = float(probabilities[1])  # Probabilidade da classe positiva
            except:
                probability = None
            
            return PredictionResponse(
                prediction=int(prediction),
                prediction_label='Bebe' if prediction == 1 else 'Não Bebe',
                probability=probability
            )
            
        except Exception as e:
            logger.error(f"Erro na predição: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erro na predição: {str(e)}"
            )
    
    def predict_batch(self, patients_data: List[PatientData]) -> List[PredictionResponse]:
        """Faz predições para múltiplos pacientes"""
        self._validate_service_ready()
        
        try:
            # Converter para lista de dicionários
            data_dicts = [patient.model_dump() for patient in patients_data]
            
            # Criar DataFrame com ordem correta
            original_order = [
                'sex', 'age', 'height', 'weight', 'waistline', 'sight_left', 'sight_right',
                'hear_left', 'hear_right', 'SBP', 'DBP', 'BLDS', 'tot_chole', 'HDL_chole',
                'LDL_chole', 'triglyceride', 'hemoglobin', 'urine_protein', 'serum_creatinine',
                'SGOT_AST', 'SGOT_ALT', 'gamma_GTP', 'SMK_stat_type_cd'
            ]
            
            ordered_data_list = []
            for data_dict in data_dicts:
                ordered_data = {}
                for col in original_order:
                    if col in data_dict:
                        ordered_data[col] = data_dict[col]
                ordered_data_list.append(ordered_data)
            
            df = pd.DataFrame(ordered_data_list)
            
            # Preprocessar dados
            df_processed = self.preprocessor.process_data(df, is_training=False)
            
            # Fazer predições
            predictions = self.model.predict(df_processed)
            
            # Obter probabilidades se disponível
            try:
                probabilities = self.model.predict_proba(df_processed)
                probs = probabilities[:, 1]  # Probabilidade da classe positiva
            except:
                probs = [None] * len(predictions)
            
            results = []
            for i, prediction in enumerate(predictions):
                result = PredictionResponse(
                    prediction=int(prediction),
                    prediction_label='Bebe' if prediction == 1 else 'Não Bebe',
                    probability=float(probs[i]) if probs[i] is not None else None
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Erro na predição em lote: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erro na predição em lote: {str(e)}"
            )
    
    def get_model_info(self) -> dict:
        """Retorna informações sobre o modelo carregado"""
        if not self.model_loaded:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Modelo não carregado"
            )
        
        return {
            "model_loaded": self.model_loaded,
            "features_trained": self.features_trained,
            "model_type": str(type(self.model).__name__)
        }


# Instância global do serviço
prediction_service = PredictionService() 