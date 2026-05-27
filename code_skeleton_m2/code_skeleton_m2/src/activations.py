import numpy as np


class Sigmoid:
    @staticmethod
    def forward(z):
        return 1/(1+ np.exp(-z))

    @staticmethod
    def gradient(z):
        return Sigmoid.forward(z)*(1 - Sigmoid.forward(z))

class ReLU:
    @staticmethod
    def forward(z):
        return np.maximum(0,z)

    @staticmethod
    def gradient(z):
        return np.where(z < 0,0,1)

class SoftMax:
    @staticmethod
    def forward(z):
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    @staticmethod
    def gradient(z):
        return np.ones_like(z)