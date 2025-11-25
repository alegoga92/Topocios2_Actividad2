## 🧠 Título del Proyecto

# Análisis Predictivo del Periodo de Diagnóstico de Cáncer (WIDS Datathon 2024)

> Repositorio de Machine Learning modular para la clasificación del tiempo de diagnóstico de cáncer de mama (DiagPeriodL90D).

---

## 🎯 Objetivo del Proyecto

El objetivo principal de este proyecto es construir y evaluar un **modelo de clasificación binaria** capaz de predecir si un paciente recibirá un diagnóstico de cáncer de mama **rápido** (en menos de 90 días, `DiagPeriodL90D` = 1) o **lento** (en 90 días o más, `DiagPeriodL90D` = 0), basándose en datos sociodemográficos, clínicos y medioambientales.

La métrica de evaluación principal es el **área bajo la curva ROC (ROC-AUC)**, medida mediante validación cruzada.

---

## 🏗️ Arquitectura del Proyecto (Estructura Modular)

Este repositorio sigue una arquitectura modular para separar claramente las responsabilidades, aplicar buenas prácticas de desarrollo y facilitar la trazabilidad con MLflow.

Topicos2_Actividad2/
├── data/           # Contiene los datasets originales (training.csv, test.csv)
├── src/            # Módulos Python con la lógica de negocio
│   ├── main.py     # Orquesta el pipeline completo.
│   ├── module_data.py # Clase para carga, Feature Engineering y preprocesamiento (ColumnTransformer).
│   ├── module_ml.py   # Clase para entrenamiento, evaluación (CV) y logging con MLflow.
│   └── module_path.py # Utilitario para manejo de rutas de archivos.
└── mlruns/         # Directorio de tracking de MLflow.
└── README.md

## ⚙️ Pipeline de Preprocesamiento (`module_data.py`)

El módulo `module_data.py` ejecuta un **pipeline consistente y coherente** en ambos datasets (Training y Test) para asegurar la compatibilidad dimensional y prevenir la fuga de datos (data leakage).

### 1. Ingeniería de Características (Feature Engineering)
* **BMI:** La variable numérica `bmi` (con alta tasa de nulos) fue transformada en la variable categórica **`bmi_category`** (ej., `Normal`, `Unknown`). Esto permite capturar el valor predictivo de la falta de información.
* **Imputación Categórica Explícita:** Los nulos en `patient_race` y `payer_type` fueron imputados con la categoría **'Unknown'** o **'missing'**.
* **Eliminación:** Se eliminó la columna numérica original `bmi` y otras columnas de ID/descripción irrelevantes.

### 2. Transformaciones de Consistencia (ColumnTransformer)
Se utiliza un **`ColumnTransformer`** (entrenado solo con el set de entrenamiento) para aplicar las siguientes reglas:
* **Variables Numéricas:** Imputación de nulos por la media (`SimpleImputer`) seguida de **Estandarización (`StandardScaler`)**.
* **Variables Categóricas:** **One-Hot Encoding** con la configuración clave **`handle_unknown='ignore'`** para manejar de forma segura las categorías presentes solo en el set de prueba y mantener las 237 características consistentes.

---

## 🧪 Resultados de la Experimentación (MLflow Tracking)

Se evaluaron 6 modelos de clasificación utilizando **Validación Cruzada (K=5 folds)**, y se compararon en métricas de rendimiento y eficiencia.

| Modelo | ROC-AUC (Promedio CV) | Desv. Estándar | Tiempo (Segundos) | Conclusiones |
| :--- | :--- | :--- | :--- | :--- |
| **Regresión Logística** | **0.7946** | 0.0046 | 5.30 | **Mejor Rendimiento**. Alto poder predictivo lineal. |
| Random Forest | 0.7828 | 0.0077 | 21.10 | Sólido rendimiento, pero mayor coste computacional. |
| GaussianNB (Naive Bayes) | 0.7511 | 0.0042 | **0.49** | **Más Eficiente**. Excelente velocidad para un rendimiento aceptable. |
| Árbol de Decisión | 0.7503 | 0.0088 | 4.23 | Base de rendimiento no lineal. |
| MLPClassifier | 0.7388 | 0.0068 | 97.90 | Rendimiento bajo para su complejidad y tiempo de cómputo. |
| KNeighborsClassifier | 0.6108 | 0.0059 | 6.62 | El rendimiento más bajo; sugiere que las clases no son localmente separables. |

### Conclusiones de Modelado

1.  **Modelo Base Ganador:** La **Regresión Logística** es el modelo base con el mejor balance ROC-AUC/eficiencia, lo que sugiere que la relación entre las variables y el diagnóstico es predominantemente lineal.
2.  **Siguiente Paso:** Se requiere el **ajuste fino de hiperparámetros (Tuning)**, enfocado en el Random Forest y la Regresión Logística, para mejorar el ROC-AUC y optimizar la predicción final.

---

## 🚀 Cómo Ejecutar el Proyecto

### 1. Requisitos de Librerías

Instala todas las dependencias del proyecto:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn mlflow