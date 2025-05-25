from abc import ABC, abstractmethod
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, Tuple
import mlflow
import mlflow.sklearn
import os


# Abstract Factory para criação de modelos
class ModelFactory(ABC):
    """Factory abstrata para criação de modelos"""
    
    @abstractmethod
    def create_model(self, **params) -> 'BaseModel':
        """Cria um modelo específico"""
        pass


class RandomForestFactory(ModelFactory):
    """Factory para criação de modelos Random Forest"""
    
    def create_model(self, **params) -> 'RandomForestModel':
        return RandomForestModel(**params)


class LogisticRegressionFactory(ModelFactory):
    """Factory para criação de modelos Logistic Regression"""
    
    def create_model(self, **params) -> 'LogisticRegressionModel':
        return LogisticRegressionModel(**params)


class SVMFactory(ModelFactory):
    """Factory para criação de modelos SVM"""
    
    def create_model(self, **params) -> 'SVMModel':
        return SVMModel(**params)


# Classe base para modelos
class BaseModel(ABC):
    """Classe base abstrata para todos os modelos"""
    
    def __init__(self, **params):
        self.params = params
        self.model = None
        self.is_trained = False
        self.metrics = {}
    
    @abstractmethod
    def _create_model(self):
        """Cria a instância do modelo específico"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Retorna o nome do modelo"""
        pass
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Treina o modelo
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
        """
        if self.model is None:
            self.model = self._create_model()
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz predições
        
        Args:
            X: Features para predição
            
        Returns:
            Array com as predições
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz predições de probabilidade
        
        Args:
            X: Features para predição
            
        Returns:
            Array com as probabilidades
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda.")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("Modelo não suporta predição de probabilidade.")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Avalia o modelo
        
        Args:
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Dicionário com métricas
        """
        y_pred = self.predict(X_test)
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        return self.metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Salva o modelo treinado
        
        Args:
            filepath: Caminho para salvar o modelo
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda.")
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(self.model, filepath)
        print(f"Modelo salvo em {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Carrega um modelo salvo
        
        Args:
            filepath: Caminho do modelo salvo
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Modelo carregado de {filepath}")


# Implementações específicas dos modelos
class RandomForestModel(BaseModel):
    """Modelo Random Forest"""
    
    def _create_model(self):
        return RandomForestClassifier(**self.params)
    
    def get_model_name(self) -> str:
        return "RandomForest"


class LogisticRegressionModel(BaseModel):
    """Modelo Logistic Regression"""
    
    def _create_model(self):
        return LogisticRegression(**self.params)
    
    def get_model_name(self) -> str:
        return "LogisticRegression"


class SVMModel(BaseModel):
    """Modelo SVM"""
    
    def _create_model(self):
        return SVC(**self.params)
    
    def get_model_name(self) -> str:
        return "SVM"


# Gerenciador de modelos
class ModelManager:
    """Gerenciador para criação e manipulação de modelos"""
    
    def __init__(self):
        self.factories = {
            'random_forest': RandomForestFactory(),
            'logistic_regression': LogisticRegressionFactory(),
            'svm': SVMFactory()
        }
    
    def create_model(self, model_type: str, **params) -> BaseModel:
        """
        Cria um modelo usando a factory apropriada
        
        Args:
            model_type: Tipo do modelo
            **params: Parâmetros do modelo
            
        Returns:
            Instância do modelo
        """
        if model_type not in self.factories:
            raise ValueError(f"Tipo de modelo '{model_type}' não suportado. "
                           f"Tipos disponíveis: {list(self.factories.keys())}")
        
        return self.factories[model_type].create_model(**params)
    
    def train_and_evaluate(self, model: BaseModel, X_train: pd.DataFrame, 
                          y_train: pd.Series, X_test: pd.DataFrame, 
                          y_test: pd.Series) -> Dict[str, float]:
        """
        Treina e avalia um modelo
        
        Args:
            model: Instância do modelo
            X_train: Features de treino
            y_train: Target de treino
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Dicionário com métricas
        """
        # Treinar modelo
        model.train(X_train, y_train)
        
        # Avaliar modelo
        metrics = model.evaluate(X_test, y_test)
        
        return metrics
    
    def log_to_mlflow(self, model: BaseModel, metrics: Dict[str, float], 
                     X_train: pd.DataFrame, experiment_name: str = "ML Experiment") -> None:
        """
        Registra modelo e métricas no MLflow
        
        Args:
            model: Modelo treinado
            metrics: Métricas do modelo
            X_train: Features de treino (para signature)
            experiment_name: Nome do experimento
        """
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            # Log dos parâmetros
            mlflow.log_params(model.params)
            
            # Log das métricas
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Set tag
            mlflow.set_tag("Model Type", model.get_model_name())
            
            # Infer signature
            signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
            
            # Log do modelo
            model_info = mlflow.sklearn.log_model(
                sk_model=model.model,
                artifact_path=f"{model.get_model_name()}_model",
                signature=signature,
                input_example=X_train.head(1),
                registered_model_name=f"{model.get_model_name()}-drink-prediction"
            )
            
            print(f"Modelo registrado no MLflow: {model_info.model_uri}")


# Pipeline completo de modelagem
class ModelingPipeline:
    """Pipeline completo para treinamento e avaliação de modelos"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.trained_models = {}
    
    def run_experiment(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_test: pd.DataFrame, y_test: pd.Series,
                      model_configs: Dict[str, Dict[str, Any]],
                      experiment_name: str = "Drinking Prediction") -> Dict[str, Dict[str, float]]:
        """
        Executa experimento com múltiplos modelos
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_test: Features de teste
            y_test: Target de teste
            model_configs: Configurações dos modelos
            experiment_name: Nome do experimento
            
        Returns:
            Dicionário com resultados de todos os modelos
        """
        results = {}
        
        for model_name, config in model_configs.items():
            print(f"\nTreinando modelo: {model_name}")
            
            # Criar modelo
            model = self.model_manager.create_model(
                model_type=config['type'],
                **config.get('params', {})
            )
            
            # Treinar e avaliar
            metrics = self.model_manager.train_and_evaluate(
                model, X_train, y_train, X_test, y_test
            )
            
            # Salvar modelo
            model_path = f"Modelo/Artefatos/{model_name}_modelo.bin"
            model.save_model(model_path)
            
            # Log no MLflow
            self.model_manager.log_to_mlflow(model, metrics, X_train, experiment_name)
            
            # Armazenar resultados
            results[model_name] = metrics
            self.trained_models[model_name] = model
            
            print(f"Métricas para {model_name}: {metrics}")
        
        return results
    
    def get_best_model(self, results: Dict[str, Dict[str, float]], 
                      metric: str = 'f1') -> Tuple[str, BaseModel]:
        """
        Retorna o melhor modelo baseado em uma métrica
        
        Args:
            results: Resultados dos modelos
            metric: Métrica para comparação
            
        Returns:
            Tupla com nome e instância do melhor modelo
        """
        best_model_name = max(results.keys(), key=lambda x: results[x][metric])
        best_model = self.trained_models[best_model_name]
        
        return best_model_name, best_model 