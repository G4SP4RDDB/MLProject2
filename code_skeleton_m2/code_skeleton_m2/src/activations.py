import numpy as np


class Sigmoid:
    @staticmethod
    def forward(z):
        return 1/(1+ np.exp(-z))

    @staticmethod
    def gradient(z):
        return np.exp(-z)/((1+np.exp(-z))**2)

class ReLU:
    @staticmethod
    def forward(z):
        return np.maximum(0,z)

    @staticmethod
    def gradient(z):
        return np.where(z < 0,0,1)
