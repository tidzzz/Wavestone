import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_and_process_data
from model import PermissionPredictor

# 1. Configuración
EPOCHS = 50           # Cuántas veces repasamos todos los datos
LEARNING_RATE = 0.001 # Qué tan rápido aprende (muy alto = inestable, muy bajo = lento)

def train_model():
    # --- A. CARGAR DATOS ---
    data = load_and_process_data()
    X_train = data["X_train"]
    y_train = data["y_train"]
    
    # Obtener dimensiones automáticamente
    input_dim = X_train.shape[1]  # 56
    output_dim = y_train.shape[1] # 251
    
    print(f"\n--- Iniciando Entrenamiento ---")
    print(f"Entrada: {input_dim} -> Salida: {output_dim}")

    # --- B. INICIALIZAR MODELO ---
    model = PermissionPredictor(input_dim, output_dim)
    
    # Función de Pérdida (Loss Function)
    # BCELoss = Binary Cross Entropy. Es la estándar para comparar (0, 1) vs Probabilidad.
    criterion = nn.BCELoss()
    
    # Optimizador (El que ajusta los pesos)
    # Adam es el mejor "todoterreno" hoy en día.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- C. BUCLE DE ENTRENAMIENTO ---
    model.train() # Poner modo entrenamiento (activa el Dropout)
    
    for epoch in range(EPOCHS):
        # 1. Resetear gradientes (siempre se hace al inicio del paso)
        optimizer.zero_grad()
        
        # 2. Predicción (Forward Pass)
        outputs = model(X_train)
        
        # 3. Calcular el error (Loss)
        # Compara lo que predijo (outputs) con la realidad (y_train)
        loss = criterion(outputs, y_train)
        
        # 4. Aprendizaje (Backward Pass)
        loss.backward()  # Calcula qué neuronas tuvieron la culpa del error
        optimizer.step() # Ajusta los pesos para reducir el error
        
        # Reportar progreso cada 10 épocas
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

    print("\n¡Entrenamiento finalizado!")
    
    # --- D. GUARDAR EL CEREBRO ---
    torch.save(model.state_dict(), "iam_model.pth")
    print("Modelo guardado como 'iam_model.pth'")

if __name__ == "__main__":
    train_model()