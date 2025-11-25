# Librería estándar
import os
import numpy as np
import pandas as pd
from typing import Tuple

# Scikit-learn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split # Añadido para posible división interna

# Módulos propios
from module_path import train_data_path, test_data_path

COL_PATIENT_ID = "patient_id"
COL_PATIENT_RACE = "patient_race"
COL_PAYER_TYPE = "payer_type"
COL_PATIENT_STATE = "patient_state"
COL_PATIENT_ZIP3 = "patient_zip3"
COL_PATIENT_AGE = "patient_age"
COL_PATIENT_GENDER = "patient_gender"
COL_BMI = "bmi"
COL_BREAST_CANCER_DIAGNOSIS_CODE = "breast_cancer_diagnosis_code"
COL_BREAST_CANCER_DIAGNOSIS_DESC = "breast_cancer_diagnosis_desc"
COL_METASTATIC_CANCER_DIAGNOSIS_CODE = "metastatic_cancer_diagnosis_code"
COL_METASTATIC_FIRST_NOVEL_TREATMENT = "metastatic_first_novel_treatment"
COL_METASTATIC_FIRST_NOVEL_TREATMENT_TYPE = "metastatic_first_novel_treatment_type"
COL_REGION = "Region"
COL_DIVISION = "Division"
COL_POPULATION = "population"
COL_DENSITY = "density"
COL_AGE_MEDIAN = "age_median"
COL_AGE_UNDER_10 = "age_under_10"
COL_AGE_10_TO_19 = "age_10_to_19"
COL_AGE_20S = "age_20s"
COL_AGE_30S = "age_30s"
COL_AGE_40S = "age_40s"
COL_AGE_50S = "age_50s"
COL_AGE_60S = "age_60s"
COL_AGE_70S = "age_70s"
COL_AGE_OVER_80 = "age_over_80"
COL_MALE = "male"
COL_FEMALE = "female"
COL_MARRIED = "married"
COL_DIVORCED = "divorced"
COL_NEVER_MARRIED = "never_married"
COL_WIDOWED = "widowed"
COL_FAMILY_SIZE = "family_size"
COL_FAMILY_DUAL_INCOME = "family_dual_income"
COL_INCOME_HOUSEHOLD_MEDIAN = "income_household_median"
COL_INCOME_HOUSEHOLD_UNDER_5 = "income_household_under_5"
COL_INCOME_HOUSEHOLD_5_TO_10 = "income_household_5_to_10"
COL_INCOME_HOUSEHOLD_10_TO_15 = "income_household_10_to_15"
COL_INCOME_HOUSEHOLD_15_TO_20 = "income_household_15_to_20"
COL_INCOME_HOUSEHOLD_20_TO_25 = "income_household_20_to_25"
COL_INCOME_HOUSEHOLD_25_TO_35 = "income_household_25_to_35"
COL_INCOME_HOUSEHOLD_35_TO_50 = "income_household_35_to_50"
COL_INCOME_HOUSEHOLD_50_TO_75 = "income_household_50_to_75"
COL_INCOME_HOUSEHOLD_75_TO_100 = "income_household_75_to_100"
COL_INCOME_HOUSEHOLD_100_TO_150 = "income_household_100_to_150"
COL_INCOME_HOUSEHOLD_150_OVER = "income_household_150_over"
COL_INCOME_HOUSEHOLD_SIX_FIGURE = "income_household_six_figure"
COL_INCOME_INDIVIDUAL_MEDIAN = "income_individual_median"
COL_HOME_OWNERSHIP = "home_ownership"
COL_HOUSING_UNITS = "housing_units"
COL_HOME_VALUE = "home_value"
COL_RENT_MEDIAN = "rent_median"
COL_RENT_BURDEN = "rent_burden"
COL_EDUCATION_LESS_HIGHSCHOOL = "education_less_highschool"
COL_EDUCATION_HIGHSCHOOL = "education_highschool"
COL_EDUCATION_SOME_COLLEGE = "education_some_college"
COL_EDUCATION_BACHELORS = "education_bachelors"
COL_EDUCATION_GRADUATE = "education_graduate"
COL_EDUCATION_COLLEGE_OR_ABOVE = "education_college_or_above"
COL_EDUCATION_STEM_DEGREE = "education_stem_degree"
COL_LABOR_FORCE_PARTICIPATION = "labor_force_participation"
COL_UNEMPLOYMENT_RATE = "unemployment_rate"
COL_SELF_EMPLOYED = "self_employed"
COL_FARMER = "farmer"
COL_RACE_WHITE = "race_white"
COL_RACE_BLACK = "race_black"
COL_RACE_ASIAN = "race_asian"
COL_RACE_NATIVE = "race_native"
COL_RACE_PACIFIC = "race_pacific"
COL_RACE_OTHER = "race_other"
COL_RACE_MULTIPLE = "race_multiple"
COL_HISPANIC = "hispanic"
COL_DISABLED = "disabled"
COL_POVERTY = "poverty"
COL_LIMITED_ENGLISH = "limited_english"
COL_COMMUTE_TIME = "commute_time"
COL_HEALTH_UNINSURED = "health_uninsured"
COL_VETERAN = "veteran"
COL_OZONE = "Ozone"
COL_PM25 = "PM25"
COL_N02 = "N02"
COL_DIAGPERIODL90D = "DiagPeriodL90D"

# Columnas que se eliminarán
COLS_TO_DROP = [
    "patient_id",
    "breast_cancer_diagnosis_desc",
    "metastatic_first_novel_treatment",
    "metastatic_first_novel_treatment_type"
]

class Dataset():

    def __init__(self, num_samples:int=None, seed:int=100):
        self.num_samples = num_samples
        self.seed = seed
        self.preprocessor = None # Almacenará el ColumnTransformer ajustado

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        train_path = train_data_path()
        test_path = test_data_path()

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        # Eliminar columnas con poca o nula información predictiva
        df_train = df_train.drop(columns=COLS_TO_DROP)
        df_test = df_test.drop(columns=COLS_TO_DROP)

        if self.num_samples is not None:
            df_train = df_train.sample(n=self.num_samples, random_state=self.seed)
            df_test = df_test.sample(n=self.num_samples, random_state=self.seed)

        return df_train, df_test
    
    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # --- 1. Ingeniería de Características: BMI ---
        # Crea bmi_category y elimina bmi original (69% nulos)
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf], 
                                    labels=['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II', 'Extreme']).astype(object)
        df['bmi_category'] = df['bmi_category'].fillna('Unknown')
        df = df.drop(columns=['bmi'])

        # --- 2. Imputación explícita (para NULOS ALTOS) ---
        # Imputa nulos en columnas categóricas de alta tasa de pérdida (49% en patient_race)
        df['patient_race'] = df['patient_race'].fillna('Unknown')
        df['payer_type'] = df['payer_type'].fillna('Unknown')
        
        # Otras imputaciones menores se harán en el pipeline (paso 3)

        return df

    def get_preprocessor(self, X_train: pd.DataFrame, method: str = 'std') -> ColumnTransformer:
        
        # 1. Definir columnas numéricas y categóricas
        train_numeric = X_train.select_dtypes(include=['number']).columns.tolist()
        train_categories = X_train.select_dtypes(include=['object']).columns.tolist()

        # 2. Definir el escalador (StandardScaler o MinMaxScaler)
        if method == 'std':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'std' or 'minmax'")

        # 3. Pipeline para variables numéricas (Imputación con media + Escalado)
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', scaler)
        ])

        # 4. Pipeline para variables categóricas (Imputación con constante + OneHotEncoding)
        categorical_transformer = Pipeline(steps=[
            # Imputación de nulos restantes (ej. si quedan nulos tras la ingeniería)
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing_cat')),
            # OHE con manejo de inconsistencias de bases de datos
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first', dtype=int))
        ])

        # 5. ColumnTransformer: Combina los pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, train_numeric),
                ('cat', categorical_transformer, train_categories)
            ],
            # remainder='passthrough' si hubiera columnas que NO queremos transformar
            remainder='drop' 
        )
        
        return preprocessor, train_numeric, train_categories

    
    def load_xy_final(self, method: str = 'std') -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        
        # --- 1. Cargar y aplicar Ingeniería de Características ---
        df_train_raw, df_test_raw = self.load_data()
        
        df_train = self.apply_feature_engineering(df_train_raw.copy())
        df_test = self.apply_feature_engineering(df_test_raw.copy())

        # --- 2. Separar X e Y (Target) ---
        X_train = df_train.drop(columns=[COL_DIAGPERIODL90D])
        y_train = df_train[COL_DIAGPERIODL90D]
        X_test = df_test # df_test ya no tiene el Target

        # --- 3. Definir y Entrenar el Preprocesador (ColumnTransformer) ---
        self.preprocessor, train_numeric, train_categories = self.get_preprocessor(X_train, method)
        
        # Ajustar (FIT) el preprocesador SÓLO con los datos de ENTRENAMIENTO
        self.preprocessor.fit(X_train)

        # --- 4. Transformar AMBOS sets ---
        X_train_encoded = self.preprocessor.transform(X_train)
        X_test_encoded = self.preprocessor.transform(X_test)

        # --- 5. Reconstruir los DataFrames ---
        
        # Obtener los nombres de las columnas resultantes
        encoded_col_names = self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(train_categories)
        final_columns = list(encoded_col_names) + train_numeric 

        # Crear los DataFrames finales
        X_train_final = pd.DataFrame(X_train_encoded, columns=final_columns)
        X_test_final = pd.DataFrame(X_test_encoded, columns=final_columns)

        # Asegurar que el índice de y_train coincida con X_train_final si se requiere
        y_train.index = X_train_final.index
        
        print(f"Shape X_train_final: {X_train_final.shape}")
        print(f"Shape X_test_final: {X_test_final.shape}")
        
        return X_train_final, y_train, X_test_final

    
    def load_xy(self, method:str='std'):
        """ Método llamado por main.py """
        X_train, y_train, X_test = self.load_xy_final(method=method)

        # Se retorna X_train y y_train para el entrenamiento del modelo
        return X_train, y_train