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

    def linear(self, x, W, b):
        return x @ W + b

    def forward(self, x):
        z1 = self.linear(x, self.W1, self.b1)
        a1 = self.relu(z1)
        z2 = self.linear(a1, self.W2, self.b2)
        return z2
