# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB # Añadido el modelo Naive Bayes

# Módulos propios
from module_data import Dataset # class Dataset
from module_ml import Model


def main():

    # 1. Carga y Preprocesamiento de Datos
    # Se inicializa la clase Dataset, que ahora contiene toda la lógica de limpieza y ColumnTransformer
    data = Dataset() 
    
    # load_xy retorna X_train y y_train, ya limpios, escalados y codificados
    # Usamos 'std' (StandardScaler) como método de escalado por defecto
    X, y = data.load_xy(method='std') 
    
    print(f"\n--- Preparación de Datos Completada ---")
    print(f"Shape X_train para modelado: {X.shape}")
    print(f"Shape y_train para modelado: {y.shape}")
    print("--------------------------------------")
    
    # 2. Inicialización del Módulo de Modelado
    # La clase Model se encarga de la validación cruzada y el tracking con MLflow.
    ml = Model(X=X, y=y, seed=42)
    
    # 3. Experimentación con Modelos (Al menos 4 requeridos)
    
    # Modelo 1: Regresión Logística
    # Aumentamos max_iter para asegurar convergencia con datos escalados.
    ml.evaluate(LogisticRegression(max_iter=5000, random_state=42))
    
    # Modelo 2: K-Vecinos más Cercanos (KNN)
    ml.evaluate(KNeighborsClassifier(n_neighbors=5))
    
    # Modelo 3: Árbol de Decisión
    ml.evaluate(DecisionTreeClassifier(max_depth=10, random_state=42))
    
    # Modelo 4: Random Forest (Modelo de ensamble más complejo)
    ml.evaluate(RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))

    # Modelo 5: Naive Bayes (Modelo simple, bueno para datos de alta dimensión)
    ml.evaluate(GaussianNB())

    # Modelo 6: Red Neuronal Perceptrón Multicapa (MLP)
    ml.evaluate(MLPClassifier(random_state=42, max_iter=2000, hidden_layer_sizes=(100, 50), activation='relu'))
    
    # El módulo module_ml registrará automáticamente las métricas (ROC-AUC, tiempo) 
    # y los parámetros para cada uno de estos runs en la carpeta mlruns/.

if __name__ == "__main__":
    main()