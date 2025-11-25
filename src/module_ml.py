# Librerías
import pandas as pd
import numpy as np
import time
from typing import Dict, Any

# scikit-learn
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.base import ClassifierMixin # Usado para tipado

# mlflow
import mlflow
from mlflow.models.signature import infer_signature

# ====================================================================
# CONFIGURACIÓN DE MLFLOW Y DECORADOR
# ====================================================================

def mlflow_logger(func):
    """Decorador para configurar y ejecutar un run de MLflow."""
    def wrapper(*args, **kwargs):
        # La ruta mlruns_path debe ser accesible desde la ejecución
        mlruns_path = "../mlruns"
        mlflow.set_tracking_uri(mlruns_path)
        experiment_name = "WIDS2024"

        try:
            # Crea un nuevo experimento si no existe
            exp_id = mlflow.create_experiment(name=experiment_name)
        except Exception:
            # Si ya existe, obtiene su ID
            exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            
        with mlflow.start_run(experiment_id=exp_id):
            return func(*args, **kwargs)
    return wrapper


# ====================================================================
# CLASE MODEL
# ====================================================================

class Model():

    def __init__(self, X:pd.DataFrame, y:pd.Series, seed:int=42):
        self.X = X
        self.y = y
        self.seed = seed
        # Configuración de la validación cruzada (5 folds)
        self.cv = KFold(n_splits=5, shuffle=True, random_state=self.seed)

    @mlflow_logger # Este decorador inicia un run de MLflow para cada modelo
    def evaluate(self, model: ClassifierMixin):
        
        model_name = type(model).__name__
        print(f"--- Evaluando Modelo: {model_name} ---")

        # 1. Evaluación con Cross-Validation
        # Usamos 'roc_auc' como métrica principal, ya que es la más común en Datathons.
        start_time = time.time()
        
        # cross_val_score entrena el modelo 5 veces (KFold)
        scores = cross_val_score(model, self.X, self.y, cv=self.cv, scoring='roc_auc', n_jobs=-1)
        
        end_time = time.time()
        
        # 2. Métricas promedio y tiempo
        avg_roc_auc = scores.mean()
        std_roc_auc = scores.std()
        time_elapsed = end_time - start_time
        
        print(f"ROC-AUC (Promedio CV): {avg_roc_auc:.4f}")
        print(f"Desviación Estándar ROC-AUC: {std_roc_auc:.4f}")
        print(f"Tiempo de cómputo (CV): {time_elapsed:.2f} segundos")
        print("-" * 30)

        # 3. Registro con MLflow
        
        # a) Parámetros
        mlflow.log_param("model_type", model_name)
        # Registra todos los parámetros del modelo (ej. n_estimators, max_depth)
        for param, value in model.get_params().items():
            # Evita registrar objetos complejos si no son relevantes para el tuning
            if isinstance(value, (int, float, str, bool, type(None))):
                mlflow.log_param(param, value)
            
        # b) Métricas
        mlflow.log_metric("ROC-AUC_avg", avg_roc_auc)
        mlflow.log_metric("ROC-AUC_std", std_roc_auc)
        mlflow.log_metric("runtime_seconds", time_elapsed)

        # c) Loguear el modelo (Ajuste simple para generar el artefacto)
        model.fit(self.X, self.y)
        # Se infiere la firma del modelo para su posterior despliegue
        signature = infer_signature(self.X, model.predict(self.X))
        mlflow.sklearn.log_model(model, "model", signature=signature)
        
        return avg_roc_auc