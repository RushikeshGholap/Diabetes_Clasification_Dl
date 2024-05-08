import numpy as np
from Layers.Layer import Layer


class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_in):
        shift_x = data_in - np.max(data_in, axis=-1, keepdims=True)
        exps = np.exp(shift_x)
        self.softmax_output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.softmax_output

    def gradient(self):
        y = self.getPrevOut()
        return y * (1 - y)  

    def backward(self, grad_in):
        return grad_in 
