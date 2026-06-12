import numpy as np
from model import PolynomialNeuralNetwork

def polynomial(x): 
    return 2*x**3 - 1.5*x**2 + 0.5*x + 1

def generate_polynomial_data(n_samples=300, noise_std=0.15, x_range=(-2, 2)):

    np.random.seed(42)
    X = np.random.uniform(x_range[0], x_range[1], (n_samples, 1))
    y = polynomial(X) + np.random.normal(0, noise_std, (n_samples, 1))
    return X, y

def train():
    
    X_train, y_train = generate_polynomial_data()
    model = PolynomialNeuralNetwork(input_size=1, hidden_size=20, output_size=1)
    
    print("Model training...")
    model.fit(X_train, y_train, epochs=5000, lr=0.02, verbose=True)
    return model