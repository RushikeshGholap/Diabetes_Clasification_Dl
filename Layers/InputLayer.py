import numpy as np
from Layers.Layer import Layer

class InputLayer(Layer):
    def __init__(self, input_data):
        if not isinstance(input_data, np.ndarray):
            raise TypeError("Input data must be a NumPy array.")
        if input_data.ndim != 2:
            raise ValueError("Input data must be a 2D array.")

        self.mean = np.mean(input_data, axis=0)
        self.std = np.std(input_data, axis=0)
        self.std[self.std == 0] = 1

    def forward(self, input_data):
        if not isinstance(input_data, np.ndarray):
            raise TypeError("Input data must be a NumPy array.")
        if input_data.shape[1] != self.mean.shape[0]:
            raise ValueError("Input data must have the same number of features as the training data.")

        return (input_data - self.mean) / self.std

    def gradient(self):
        pass

    def backward(self, grad_input):
        pass
