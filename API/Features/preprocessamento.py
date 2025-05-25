import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler
from typing import Dict, Any
import numpy as np
import os


class DataPreprocessor:
    """Classe responsável pelo pré-processamento dos dados"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.ordinal_encoder = OrdinalEncoder()
        self.is_fitted = False
        self.column_order = None  # Para armazenar a ordem das colunas
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa e converte os dados para os tipos apropriados
        
        Args:
            df: DataFrame com os dados brutos
            
        Returns:
            DataFrame com dados limpos
        """
        df_clean = df.copy()
        
        # Alterar variável alvo para ser 1 ou 0
        if 'DRK_YN' in df_clean.columns:
            df_clean['DRK_YN'] = df_clean['DRK_YN'].map({'Y': 1, 'N': 0})
        
        # Alterar sexo para ser 1 ou 0
        if 'sex' in df_clean.columns:
            df_clean['sex'] = df_clean['sex'].map({'Male': 1, 'Female': 0})
        
        # Converter colunas específicas para float64
        cols_to_convert = ['age', 'height', 'weight', 'DRK_YN', 'sex']
        existing_cols = [col for col in cols_to_convert if col in df_clean.columns]
        df_clean[existing_cols] = df_clean[existing_cols].astype('float64')
        
        # Converter todas as colunas float para numeric
        float_cols = df_clean.select_dtypes(include='float').columns
        df_clean[float_cols] = df_clean[float_cols].apply(pd.to_numeric, errors='coerce')
        
        return df_clean
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria novas features baseadas nas existentes
        
        Args:
            df: DataFrame com dados limpos
            
        Returns:
            DataFrame com novas features
        """
        df_featured = df.copy()
        
        # Criar IMC (Índice de Massa Corporal)
        if 'weight' in df_featured.columns and 'height' in df_featured.columns:
            df_featured['IMC'] = df_featured['weight'] / ((df_featured['height'] / 100) ** 2)
        
        # Criar feature de sobrecarga de órgãos
        if 'SGOT_AST' in df_featured.columns:
            df_featured['organ_overload'] = df_featured['SGOT_AST'].apply(
                lambda x: 1 if x > 40 else 0
            )
        
        # Criar feature de função hepática
        if 'sex' in df_featured.columns and 'gamma_GTP' in df_featured.columns:
            df_featured['fail_hepatic_func'] = df_featured.apply(
                self._categorize_hepatic_function, axis=1
            )
        
        return df_featured
    
    def _categorize_hepatic_function(self, row) -> int:
        """
        Categoriza a função hepática baseada no sexo e gamma_GTP
        
        Args:
            row: Linha do DataFrame
            
        Returns:
            1 se não saudável, 0 se saudável
        """
        if row['sex'] == 1 and row['gamma_GTP'] >= 63:
            return 1  # Não saudável
        elif row['sex'] == 0 and row['gamma_GTP'] >= 35:
            return 1  # Não saudável
        else:
            return 0  # Saudável
    
    def remove_unnecessary_columns(self, df: pd.DataFrame, columns_to_drop: list = None) -> pd.DataFrame:
        """
        Remove colunas desnecessárias após a criação de features
        
        Args:
            df: DataFrame com features
            columns_to_drop: Lista de colunas para remover
            
        Returns:
            DataFrame sem as colunas removidas
        """
        if columns_to_drop is None:
            columns_to_drop = ['height', 'weight']
        
        existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
        return df.drop(existing_cols_to_drop, axis=1)
    
    def fit_transformers(self, df: pd.DataFrame, target_column: str = 'DRK_YN') -> pd.DataFrame:
        """
        Ajusta os transformadores nos dados de treino
        
        Args:
            df: DataFrame com dados de treino
            target_column: Nome da coluna alvo
            
        Returns:
            DataFrame transformado
        """
        # Separar features e target
        X = df.drop(target_column, axis=1) if target_column in df.columns else df
        
        # Salvar a ordem das colunas
        self.column_order = list(X.columns)
        
        # Fit do scaler
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Se há target, adicionar de volta
        if target_column in df.columns:
            X_scaled_df[target_column] = df[target_column]
        
        self.is_fitted = True
        return X_scaled_df
    
    def transform_data(self, df: pd.DataFrame, target_column: str = 'DRK_YN') -> pd.DataFrame:
        """
        Transforma os dados usando transformadores já ajustados
        
        Args:
            df: DataFrame com dados para transformar
            target_column: Nome da coluna alvo
            
        Returns:
            DataFrame transformado
        """
        if not self.is_fitted:
            raise ValueError("Transformadores não foram ajustados ainda. Use fit_transformers primeiro.")
        
        # Separar features e target
        X = df.drop(target_column, axis=1) if target_column in df.columns else df
        
        # Garantir que as colunas estejam na ordem correta
        if self.column_order is not None:
            # Reordenar colunas conforme a ordem do treinamento
            X = X[self.column_order]
        
        # Transform usando scaler já ajustado
        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Se há target, adicionar de volta
        if target_column in df.columns:
            X_scaled_df[target_column] = df[target_column]
        
        return X_scaled_df
    
    def process_data(self, df: pd.DataFrame, is_training: bool = False, target_column: str = 'DRK_YN') -> pd.DataFrame:
        """
        Pipeline completo de pré-processamento
        
        Args:
            df: DataFrame com dados brutos
            is_training: Se é dados de treino (para fit dos transformadores)
            target_column: Nome da coluna alvo
            
        Returns:
            DataFrame processado
        """
        # Pipeline de pré-processamento
        df_processed = self.clean_data(df)
        df_processed = self.create_features(df_processed)
        df_processed = self.remove_unnecessary_columns(df_processed)
        
        # Aplicar transformações
        if is_training:
            df_processed = self.fit_transformers(df_processed, target_column)
        else:
            df_processed = self.transform_data(df_processed, target_column)
        
        return df_processed
    
    def save_artifacts(self, base_path: str = "Features/Artefatos"):
        """
        Salva os artefatos de pré-processamento
        
        Args:
            base_path: Caminho base para salvar os artefatos
        """
        if not self.is_fitted:
            raise ValueError("Transformadores não foram ajustados ainda.")
        
        # Criar diretório se não existir
        os.makedirs(base_path, exist_ok=True)
        
        # Salvar scaler
        with open(f"{base_path}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        
        # Salvar ordinal encoder (mesmo que não usado, para manter estrutura)
        with open(f"{base_path}/ordinal.pkl", "wb") as f:
            pickle.dump(self.ordinal_encoder, f)
        
        # Salvar ordem das colunas
        with open(f"{base_path}/column_order.pkl", "wb") as f:
            pickle.dump(self.column_order, f)
        
        print(f"Artefatos salvos em {base_path}")
    
    def load_artifacts(self, base_path: str = "Features/Artefatos"):
        """
        Carrega os artefatos de pré-processamento
        
        Args:
            base_path: Caminho base dos artefatos
        """
        # Carregar scaler
        with open(f"{base_path}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        
        # Carregar ordinal encoder
        with open(f"{base_path}/ordinal.pkl", "rb") as f:
            self.ordinal_encoder = pickle.load(f)
        
        # Carregar ordem das colunas
        try:
            with open(f"{base_path}/column_order.pkl", "rb") as f:
                self.column_order = pickle.load(f)
        except FileNotFoundError:
            print("Aviso: Arquivo column_order.pkl não encontrado. Retreine o modelo.")
            self.column_order = None
        
        self.is_fitted = True
        print(f"Artefatos carregados de {base_path}") 