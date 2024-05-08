import numpy as np
from Layers.Layer import Layer

class ReLULayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_in):
        return np.maximum(0, data_in)

    def gradient(self, data_in):
        return np.where(data_in > 0, 1, 0)

    def backward(self, grad_in, data_in):
        return grad_in * self.gradient(data_in)
