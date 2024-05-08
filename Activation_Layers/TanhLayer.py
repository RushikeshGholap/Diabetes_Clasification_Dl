import numpy as np
from Layers.Layer import Layer


class TanhLayer(Layer):
    def __init__(self):
        super().__init__()
        self.prev_out = None
    def forward(self, data_in):
        self.prev_out = np.tanh(data_in.astype(float))
        return self.prev_out

    def backward(self, grad_out):
        # Derivative of tanh: 1 - (tanh(x))^2
        return grad_out * (1 - np.square(self.prev_out))

    def gradient(self):
        y = self.getPrevOut()
        return 1 - np.square(y)
