#!/usr/bin/env python3
"""
Script para treinamento de modelos de ML
"""

import argparse
import logging
from pathlib import Path

from Features.preprocessamento import DataPreprocessor
from Modelo.modelo import ModelingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_models(data_path: str = "../smoking_drinking.parquet", test_size: float = 0.2):
    """Executa o treinamento completo dos modelos"""
    
    # Verificar se o arquivo de dados existe
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {data_path}")
    
    # Configuração dos modelos
    model_configs = {
        'random_forest': {
            'type': 'random_forest',
            'params': {'random_state': 42, 'n_estimators': 100}
        },
        'logistic_regression': {
            'type': 'logistic_regression',
            'params': {'random_state': 42, 'max_iter': 1000}
        }
    }
    
    try:
        from sklearn.model_selection import train_test_split
        import pandas as pd
        
        # Inicializar componentes
        preprocessor = DataPreprocessor()
        pipeline = ModelingPipeline()
        
        # Carregar dados
        logger.info("Carregando dados...")
        df = pd.read_parquet(data_path)
        
        # Preprocessar dados
        logger.info("Preprocessando dados...")
        df_processed = preprocessor.process_data(df, is_training=True)
        
        # Salvar artefatos de preprocessamento
        preprocessor.save_artifacts()
        
        # Preparar dados para modelagem
        X = df_processed.drop('DRK_YN', axis=1)
        y = df_processed['DRK_YN']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Executar experimentos
        logger.info("Executando experimentos...")
        results = pipeline.run_experiment(
            X_train, y_train, X_test, y_test, model_configs
        )
        
        # Obter melhor modelo
        best_model_name, best_model = pipeline.get_best_model(results)
        
        # Salvar melhor modelo como padrão
        best_model.save_model("Modelo/Artefatos/modelo.bin")
        
        logger.info(f"Treinamento concluído. Melhor modelo: {best_model_name}")
        
        return {
            'results': results,
            'best_model': best_model_name,
            'best_metrics': results[best_model_name]
        }
        
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        raise


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Treinar modelos de predição de consumo de álcool')
    parser.add_argument('--data', default='../smoking_drinking.parquet',
                       help='Caminho para o arquivo de dados')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proporção dos dados para teste (padrão: 0.2)')
    
    args = parser.parse_args()
    
    try:
        results = train_models(args.data, args.test_size)
        
        print("\n" + "="*50)
        print("RESULTADOS DO TREINAMENTO")
        print("="*50)
        
        for model_name, metrics in results['results'].items():
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        print(f"\n🏆 Melhor modelo: {results['best_model']}")
        print("✅ Artefatos salvos em Features/Artefatos/ e Modelo/Artefatos/")
        
    except Exception as e:
        print(f"❌ Erro durante o treinamento: {str(e)}")
        exit(1)


if __name__ == '__main__':
    main() 