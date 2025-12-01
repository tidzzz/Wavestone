import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
import os

def load_and_process_data(data_dir="iam_dataset"):
    print("--- Cargando datos ---")
    
    # 1. Cargar CSVs
    users_df = pd.read_csv(os.path.join(data_dir, "users.csv"))
    rights_df = pd.read_csv(os.path.join(data_dir, "rights.csv"))
    perms_df = pd.read_csv(os.path.join(data_dir, "permissions.csv"))
    
    # --- PREPARAR INPUT (X) ---
    print("Procesando atributos de usuario (X)...")
    
    # Seleccionamos las columnas que definen al usuario
    feature_cols = ["position", "department", "location", "contract_type", "seniority"]
    
    # Convertimos variables categóricas a numéricas (One-Hot Encoding)
    # Ejemplo: location_Paris: 1, location_Lyon: 0
    X_encoded = pd.get_dummies(users_df[feature_cols], dtype=float)
    
    # Guardamos los nombres de las columnas para saber qué significa cada neurona de entrada
    feature_names = X_encoded.columns.tolist()
    
    # Convertimos a matriz NumPy
    X_values = X_encoded.values
    
    # --- PREPARAR OUTPUT (Y) ---
    print("Procesando matriz de permisos (Y)...")
    
    # Necesitamos agrupar los permisos por usuario en una lista
    # Ejemplo: User 1 -> [Permiso_10, Permiso_55, Permiso_90]
    user_rights = rights_df.groupby("user_id")["permission_id"].apply(list).reset_index()
    
    # Asegurarnos de que tenemos filas para todos los usuarios (incluso si no tienen permisos)
    # Hacemos un merge con users_df para alinear el orden
    df_merged = pd.merge(users_df[["user_id"]], user_rights, on="user_id", how="left")
    
    # Rellenar NaNs con lista vacía (usuarios sin permisos)
    df_merged["permission_id"] = df_merged["permission_id"].apply(lambda x: x if isinstance(x, list) else [])
    
    # Usar MultiLabelBinarizer para crear la matriz de 0s y 1s
    # Esto crea una columna por cada permiso existente
    mlb = MultiLabelBinarizer()
    Y_matrix = mlb.fit_transform(df_merged["permission_id"])
    
    # Guardamos qué ID de permiso corresponde a cada columna
    permission_classes = mlb.classes_
    
    print(f"\nResumen de Datos:")
    print(f"Usuarios (Samples): {X_values.shape[0]}")
    print(f"Atributos de Entrada (Features): {X_values.shape[1]}")
    print(f"Total Permisos Posibles (Labels): {Y_matrix.shape[1]}")
    
    # --- SPLIT TRAIN / TEST ---
    # Dividimos 80% entrenamiento, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X_values, Y_matrix, test_size=0.2, random_state=42
    )
    
    # Convertir a Tensores de PyTorch (necesario para la red neuronal)
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    return {
        "X_train": X_train_tensor, "y_train": y_train_tensor,
        "X_test": X_test_tensor, "y_test": y_test_tensor,
        "feature_names": feature_names,
        "permission_classes": permission_classes
    }

if __name__ == "__main__":
    # Prueba rápida
    data = load_and_process_data()
    print("\n¡Datos listos para la Red Neuronal!")