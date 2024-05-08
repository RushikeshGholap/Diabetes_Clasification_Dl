import numpy as np

class CrossEntropy:
    def eval(self, Y, Yhat):
        epsilon = 1e-12
        Yhat = np.clip(Yhat, epsilon, 1. - epsilon)
        return -np.sum(Y * np.log(Yhat)) / Yhat.shape[0]

    def gradient(self, Y, Yhat):
        return Y-Yhat
