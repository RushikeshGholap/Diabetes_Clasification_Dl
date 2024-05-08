import numpy as np
from Layers.Layer import Layer

class LogisticSigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_in):
        # Ensure data_in is a numpy array
        data_in = np.array(data_in, dtype=np.float32)
        
        self.setPrevIn(data_in)
        sigmoid_output = 1 / (1 + np.exp(-data_in))
        self.setPrevOut(sigmoid_output)
        return sigmoid_output
    
    def gradient(self):
        y = self.getPrevOut()
        return y * (1 - y)

    def backward(self, grad_in):
        return grad_in * self.gradient()

