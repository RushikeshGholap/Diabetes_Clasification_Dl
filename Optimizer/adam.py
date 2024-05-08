import numpy as np
class AdamOptimizer:
    def __init__(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = layers 
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {layer: np.zeros_like(layer.weights) for layer in layers}
        self.v = {layer: np.zeros_like(layer.weights) for layer in layers}
        self.m_bias = {layer: np.zeros_like(layer.bias) for layer in layers}
        self.v_bias = {layer: np.zeros_like(layer.bias) for layer in layers}
        self.t = 0

    def update(self):
        self.t += 1
        for layer in self.layers:
            self.m[layer] = self.beta1 * self.m[layer] + (1 - self.beta1) * layer.dJdW
            self.v[layer] = self.beta2 * self.v[layer] + (1 - self.beta2) * np.square(layer.dJdW)
            m_hat = self.m[layer] / (1 - np.power(self.beta1, self.t))
            v_hat = self.v[layer] / (1 - np.power(self.beta2, self.t))
            
            layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            self.m_bias[layer] = self.beta1 * self.m_bias[layer] + (1 - self.beta1) * layer.dJdB
            self.v_bias[layer] = self.beta2 * self.v_bias[layer] + (1 - self.beta2) * np.square(layer.dJdB)
            m_hat_bias = self.m_bias[layer] / (1 - np.power(self.beta1, self.t))
            v_hat_bias = self.v_bias[layer] / (1 - np.power(self.beta2, self.t))

            layer.bias -= self.learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)
