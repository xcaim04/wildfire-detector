import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# 1. Definición del modelo (exactamente el que tú mostraste)
# ------------------------------------------------------------
class PolynomialNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------
# 2. Generar datos sintéticos a partir de un polinomio multivariado
# ------------------------------------------------------------
# Función objetivo: f(x, y, z) = 0.5*x^2 + 1.2*y + 0.8*z^3 - 0.3*x*y*z + 0.2
def polinomio_real(x, y, z):
    return 0.5 * x**2 + 1.2 * y + 0.8 * z**3 - 0.3 * x * y * z + 0.2

# Número de muestras
n_samples = 5000

# Generar valores aleatorios en un rango razonable
np.random.seed(42)
x_data = np.random.uniform(-2, 2, n_samples)
y_data = np.random.uniform(-1, 3, n_samples)
z_data = np.random.uniform(-1.5, 1.5, n_samples)

# Calcular la salida real
y_real = polinomio_real(x_data, y_data, z_data)

# Añadir un poco de ruido para simular datos reales (opcional)
ruido = np.random.normal(0, 0.05, n_samples)
y_real_noisy = y_real + ruido

# Convertir a tensores de PyTorch
X = np.column_stack((x_data, y_data, z_data))  # forma (n_samples, 3)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_real_noisy, dtype=torch.float32).view(-1, 1)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# 3. Instanciar el modelo, función de pérdida y optimizador
# ------------------------------------------------------------
input_dim = 3          # porque tenemos x, y, z
hidden_dim = 128       # neuronas en la capa oculta
output_dim = 1

modelo = PolynomialNN(input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim)
criterio = nn.MSELoss()                     # error cuadrático medio
optimizador = optim.Adam(modelo.parameters(), lr=0.001)

# ------------------------------------------------------------
# 4. Entrenamiento
# ------------------------------------------------------------
num_epochs = 300
batch_size = 64

# Para guardar la evolución de la pérdida
perdidas_entrenamiento = []
perdidas_prueba = []

for epoch in range(num_epochs):
    # Modo entrenamiento
    modelo.train()
    
    # Mezclar datos y procesar por lotes
    indices = torch.randperm(X_train.shape[0])
    for i in range(0, X_train.shape[0], batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]
        
        optimizador.zero_grad()
        predicciones = modelo(X_batch)
        loss = criterio(predicciones, y_batch)
        loss.backward()
        optimizador.step()
    
    # Evaluación en train y test cada 10 épocas
    if epoch % 10 == 0:
        modelo.eval()
        with torch.no_grad():
            pred_train = modelo(X_train)
            loss_train = criterio(pred_train, y_train)
            pred_test = modelo(X_test)
            loss_test = criterio(pred_test, y_test)
        
        perdidas_entrenamiento.append(loss_train.item())
        perdidas_prueba.append(loss_test.item())
        print(f"Época {epoch:3d} | Pérdida train: {loss_train.item():.6f} | Pérdida test: {loss_test.item():.6f}")

# ------------------------------------------------------------
# 5. Evaluación final y gráfica
# ------------------------------------------------------------
modelo.eval()
with torch.no_grad():
    predicciones_finales = modelo(X_test)
    error_final = criterio(predicciones_finales, y_test)
    print(f"\nError cuadrático medio final en test: {error_final.item():.6f}")

# Mostrar comparación entre valor real y predicción (primeros 20 ejemplos de test)
print("\nMuestra de predicciones vs valores reales:")
print("   Real    |   Predicho")
print("------------------------")
with torch.no_grad():
    for i in range(20):
        real = y_test[i].item()
        pred = modelo(X_test[i:i+1]).item()
        print(f"{real:9.4f} | {pred:9.4f}")

# Opcional: gráfica de pérdidas
plt.figure(figsize=(10,5))
plt.plot(range(0, num_epochs, 10), perdidas_entrenamiento, label='Entrenamiento')
plt.plot(range(0, num_epochs, 10), perdidas_prueba, label='Prueba')
plt.xlabel('Época')
plt.ylabel('Pérdida (MSE)')
plt.title('Evolución del error durante el entrenamiento')
plt.legend()
plt.grid(True)
plt.show()