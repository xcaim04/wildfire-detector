import numpy as np

from src.model import PolynomialNeuralNetwork
from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    x: List[float] = Field(..., description="Lista de valores x para predecir f(x)")

class PredictResponse(BaseModel):
    predictions: List[float]


def generate_training_data(n_samples=300, noise_std=0.15, x_range=(-2, 2), seed=42):
    
    polynomial = lambda x: 2*x**3 - 1.5*x**2 + 0.5*x + 1

    np.random.seed(seed)
    X = np.random.uniform(x_range[0], x_range[1], (n_samples, 1))
    y = polynomial(X) + np.random.normal(0, noise_std, (n_samples, 1))
    return X, y

def create_and_train_model():
    print("Entrenando red neuronal...")
    model = PolynomialNeuralNetwork(input_size=1, hidden_size=20, output_size=1)
    X_train, y_train = generate_training_data()
    model.fit(X_train, y_train, epochs=5000, lr=0.02, verbose=True)
    print("Training completed.")
    return model
