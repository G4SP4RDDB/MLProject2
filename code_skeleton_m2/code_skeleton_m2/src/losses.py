import numpy as np
class MSE:
    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        return np.mean(np.square(y_true - y_pred) )

    @staticmethod
    def gradient(y_true, y_pred):
        return 2 * (y_pred - y_true) / len(y_true)

class CrossEntropy:
    @staticmethod
    def loss(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    @staticmethod
    def gradient(y_true, y_pred):
        return y_pred - y_true