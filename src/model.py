import numpy as np

"""
La RED neuronal funciona, pero tiene errores en la aproximacion
no es una aproximacion perfecta, y no se si es por el modelo o por el entrenamiento.
Hay que revisar el modelo, y el entrenamiento, y ver si se puede mejorar la aproximacion.

Entrada x   Valor Real Teórico f(x)   Predicción de la Red   Error Absoluto
0.0         1.0000                     1.2600                 0.2600
0.5         1.1250                     0.8976                 0.2274
1.0         2.0000                     2.2599                 0.2599
-0.5        0.1250                     0.1877                 0.0627
2.0         12.0000                    9.9631                 2.0369

"""

class PolynomialNeuralNetwork:
    def __init__(self, input_size=1, hidden_size=128, output_size=1, learning_rate=0.005):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        # Inicialización de pesos con He initialization, que dayron me pregunto y no sabia bien.
        # es mejor para ReLU y ayuda a evitar el problema de desvanecimiento de gradiente.
        # Hay que estudiar: Por que?
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, d_loss):
        d_z2 = d_loss
        self.d_W2 = self.a1.T @ d_z2
        self.d_b2 = np.sum(d_z2, axis=0, keepdims=True)

        d_a1 = d_z2 @ self.W2.T
        
        d_z1 = d_a1 * (self.a1 > 0).astype(float)
        
        # CORRECCIÓN: Quitamos la división por 'n' que estaba duplicada aquí
        self.d_W1 = self.x.T @ d_z1
        self.d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    def update(self):
        self.W1 -= self.lr * self.d_W1
        self.b1 -= self.lr * self.d_b1
        self.W2 -= self.lr * self.d_W2
        self.b2 -= self.lr * self.d_b2

    # CORRECCIÓN CRÍTICA: Se añade 'lr=None' a los argumentos para solucionar tu TypeError
    def fit(self, X, y, epochs=20000, lr=None, verbose=True):
        # Si se pasa un lr en fit(), actualiza el learning rate de la red
        if lr is not None:
            self.lr = lr

        # Normalización de y para mejorar la estabilidad del entrenamiento
        self.y_mean = np.mean(y)
        self.y_std = np.std(y) + 1e-8
        y_norm = (y - self.y_mean) / self.y_std

        n = X.shape[0]

        for epoch in range(epochs):
            y_pred_norm = self.forward(X)

            # MSE
            loss = np.mean((y_pred_norm - y_norm) ** 2)
            
            # El gradiente de la pérdida se promedia aquí dividiendo por el número de muestras 'n'
            d_loss = 2 * (y_pred_norm - y_norm) / n
            
            self.backward(d_loss)
            self.update()
            
            if verbose and epoch % 1000 == 0:
                print(f"Epoch {epoch:6d} | Loss: {loss:.6f}")

    def predict(self, X):
        y_pred_norm = self.forward(X)
        return y_pred_norm * self.y_std + self.y_mean
