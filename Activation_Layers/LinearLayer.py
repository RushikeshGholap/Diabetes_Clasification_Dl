import numpy as np
from Layers.Layer import Layer

class LinearLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()

    def forward(self, data_in):
        return data_in

    def gradient(self):
        return 1

    def backward(self, grad_in):
        return grad_in * self.gradient()  
