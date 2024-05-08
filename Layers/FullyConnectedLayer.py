import numpy as np


class FullyConnectedLayer:
    def __init__(self, sizeIn, sizeOut):
        super().__init__()
        limit = np.sqrt(6 / (sizeIn + sizeOut))
        self.weights = np.random.uniform(-limit, limit, (sizeIn, sizeOut))
        self.bias = np.zeros((1, sizeOut))


    def forward(self, dataIn):
        if dataIn.ndim == 2:
            dataIn = dataIn.flatten()
        if dataIn.ndim == 1:
            dataIn = dataIn.reshape(1, -1)
        self.inputs = dataIn
        return np.dot(dataIn, self.weights) + self.bias

    def backward(self, gradOut):
        dJdW = np.dot(self.inputs.T, gradOut) / gradOut.shape[0]
        dJdB = np.sum(gradOut, axis=0, keepdims=True) / gradOut.shape[0]
        gradInput = np.dot(gradOut, self.weights.T)
        self.dJdW = dJdW
        self.dJdB = dJdB
        return gradInput

    def updateWeights(self, learning_rate):
        self.weights = np.array(self.weights, dtype=np.float64)
        self.dJdW = np.array(self.dJdW, dtype=np.float64)
        self.weights -= learning_rate * self.dJdW