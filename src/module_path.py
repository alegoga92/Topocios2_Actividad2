from pathlib import Path
#%%
def get_data_path(file_name: str) -> Path:
    """
    Busca la ruta del archivo de datos (training.csv o test.csv)
    subiendo un nivel desde la ubicación del script actual (src/).

    :param file_name: El nombre del archivo de datos a buscar ('training.csv' o 'test.csv').
    :return: la ruta absoluta al archivo de datos.
    :raises Exception: Si el archivo no se encuentra.
    """
    # 1. Obtenemos la ruta del script actual (module_path.py)
    # Ejemplo: C:/.../Topicos2_Actividad2/src/module_path.py
    current_script_path = Path(__file__).resolve() 
    
    # 2. Subimos un nivel (a la carpeta raíz: Topicos2_Actividad2/)
    # y luego entramos en la carpeta 'data'
    data_file = current_script_path.parent.parent / "data" / file_name

    if data_file.exists() and data_file.is_file():
        print(f"✅ Data file '{file_name}' found in: {data_file}")
        return data_file
    else:
        raise Exception(f"❌ Data not found. Checked path: {data_file.resolve()}")

def train_data_path() -> Path:
    """Retorna la ruta al archivo training.csv."""
    return get_data_path("training.csv")
        
def test_data_path() -> Path:
    """Retorna la ruta al archivo test.csv."""
    return get_data_path("test.csv")
# %%
