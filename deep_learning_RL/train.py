import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from data_loader import load_and_process_data
from model import PermissionPredictor

# --- CONFIGURACIÓN AFINADA ---
EPOCHS = 200           # Aumentamos para dejar que converja
LEARNING_RATE = 0.001 
BATCH_SIZE = 32        # Opcional (si usaras DataLoader), por ahora Full Batch está bien

def evaluate_model(model, X_test, y_test):
    """
    Función para hacer un examen real al modelo con datos que nunca ha visto.
    """
    model.eval() # Poner en modo evaluación (apaga Dropout)
    
    with torch.no_grad():
        # 1. Predecir
        predictions = model(X_test)
        
        # 2. Convertir probabilidades a 0 o 1 (Umbral 0.5)
        # Si la red dice 0.8 -> 1 (Permiso concedido)
        # Si la red dice 0.2 -> 0 (Permiso denegado)
        predicted_labels = (predictions > 0.5).float()
        
        # Convertir a numpy para usar Sklearn
        y_true_np = y_test.numpy()
        y_pred_np = predicted_labels.numpy()
        
        # 3. Calcular métricas
        # "micro": Calcula globales (útil cuando hay desbalance de clases)
        precision = precision_score(y_true_np, y_pred_np, average='micro', zero_division=0)
        recall = recall_score(y_true_np, y_pred_np, average='micro', zero_division=0)
        f1 = f1_score(y_true_np, y_pred_np, average='micro', zero_division=0)
        
        print("\n--- 📊 REPORTE DE CALIDAD DEL MODELO (TEST SET) ---")
        print(f"Precisión : {precision:.2%} (¿Cuán fiables son los permisos que doy?)")
        print(f"Recall    : {recall:.2%} (¿Cuántos permisos necesarios encontré?)")
        print(f"F1-Score  : {f1:.2%} (Balance entre ambos)")
        print("---------------------------------------------------")
        return f1

def train_model():
    # 1. Cargar datos
    data = load_and_process_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"] # Datos para el examen final
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    print(f"\n--- Iniciando Entrenamiento Mejorado (200 Épocas) ---")

    # 2. Inicializar
    model = PermissionPredictor(input_dim, output_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Bucle
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

    print("\n¡Entrenamiento finalizado!")
    
    # 4. EVALUACIÓN FINAL
    evaluate_model(model, X_test, y_test)
    
    # 5. Guardar
    torch.save(model.state_dict(), "iam_model.pth")
    print("Modelo guardado como 'iam_model.pth'")

if __name__ == "__main__":
    train_model()