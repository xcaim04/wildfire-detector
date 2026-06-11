import numpy as np

class PolynomialNeuralNetwork:
    def __init__(self, input_size=1, hidden_size=20, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x>0).astype(float)

    def linear(self, x, W, b):
        # el @ lo que hace es multiplicar las matrices, es decir, x @ W es lo mismo que np.dot(x, W)
        return x @ W + b

    def forward(self, x):
        z1 = self.linear(x, self.W1, self.b1)
        a1 = self.relu(z1)
        z2 = self.linear(a1, self.W2, self.b2)
        return z2
    
    def backward(self, grad_output):
        
        n = grad_output.shape[0]
        
        grad_z2 = grad_output
        grad_W2 = self.a1.T @ grad_z2 / n
        grad_b2 = np.sum(grad_z2, axis=0, keepdims=True) / n
        grad_a1 = grad_z2 @ self.W2.T

        grad_z1 = grad_a1 * self.relu_derivative(self.z1)
        grad_W1 = self.x.T @ grad_z1 / n
        grad_b1 = np.sum(grad_z1, axis=0, keepdims=True) / n

        self.grad_W2 = grad_W2
        self.grad_b2 = grad_b2
        self.grad_W1 = grad_W1
        self.grad_b1 = grad_b1

    def update_parameters(self, learning_rate=0.01):
        self.W1 -= learning_rate * self.grad_W1
        self.b1 -= learning_rate * self.grad_b1
        self.W2 -= learning_rate * self.grad_W2
        self.b2 -= learning_rate * self.grad_b2
    
    def fit(self, X, y, epochs=1000, lr=0.01, verbose=True):

        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = np.mean((y_pred - y) ** 2)
            grad = 2 * (y_pred - y) / X.shape[0]
            self.backward(grad)
            self.update(lr)
            if verbose and epoch % 1000 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

    def predict(self, X):
        return self.forward(X)
    
