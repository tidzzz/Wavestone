import torch
import torch.nn as nn

class PermissionPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PermissionPredictor, self).__init__()
        
        # --- ARQUITECTURA DE LA RED ---
        
        # Capa 1: Entrada -> Capa Oculta (Expandimos para buscar combinaciones)
        # De 56 características pasamos a 128 neuronas "pensantes"
        self.layer1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU() # Función de activación (enciende neuronas si la señal es fuerte)
        
        # Capa 2: Capa Oculta -> Capa Oculta (Refinamiento)
        # Bajamos a 64 para quedarnos con lo importante
        self.layer2 = nn.Linear(128, 64)
        
        # Capa de Salida: 64 -> Número de Permisos (251)
        self.output_layer = nn.Linear(64, output_dim)
        
        # Activación Final: Sigmoid
        # Transforma el número en una probabilidad entre 0 y 1.
        # Usamos Sigmoid (y no Softmax) porque es MULTI-LABEL:
        # Un usuario puede tener Permiso A (99%) Y Permiso B (90%) a la vez.
        self.sigmoid = nn.Sigmoid()
        
        # Dropout: Apaga aleatoriamente el 20% de las neuronas al entrenar
        # para evitar que el modelo memorice los datos (Overfitting).
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Así fluyen los datos por el cerebro:
        
        x = self.layer1(x)      # Entrada -> Capa 1
        x = self.relu(x)        # Activación
        x = self.dropout(x)     # Apagar algunas neuronas al azar
        
        x = self.layer2(x)      # Capa 1 -> Capa 2
        x = self.relu(x)        # Activación
        
        x = self.output_layer(x) # Capa 2 -> Salida
        x = self.sigmoid(x)      # Convertir a probabilidad (0-1)
        
        return x