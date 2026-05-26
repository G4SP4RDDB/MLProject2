class MSE:
    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        totalLoss = 0
        for i in range(len(y_true)):
            totalLoss += (y_true[i] -y_pred[i])**2
        return totalLoss/len(y_true)

    @staticmethod
    def gradient(y_true, y_pred):
        totalLoss = 0
        gradiant = []
        for i in range(len(y_true)):
            gradiant[i] =( 2 * (y_pred[i] - y_true[i]))/ len(y_true)
        return gradiant
