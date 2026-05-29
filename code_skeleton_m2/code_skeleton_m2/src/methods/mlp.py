import numpy as np


class MLP:
    def __init__(self, dimensions, activations):
        """
        :param dimensions: list of dimensions of the neural net. (input, hidden layer, ... ,hidden layer, output)
        :param activations: list of activation functions. Must contain N-1 activation function, where N = len(dimensions).

        Example of one hidden layer with
        - 2 inputs
        - 10 hidden nodes
        - 5 outputs
        layers -->    [0,        1,          2]
        ----------------------------------------
        dimensions =  (2,     10,          5)
        activations = (      Sigmoid,      Sigmoid)
        """

        self.n_layers = len(dimensions)
        self.learning_rate = None

        self.loss_function = None
        self.class_weights = None
        self.train_losses = None # for losses visualization


        #Weights and biases are initiated by index
        self.w = {}
        self.b = {}

        self.activations = {}
        for l in range(1, self.n_layers):
            self.w[l] = np.random.randn(dimensions[l - 1], dimensions[l])/ np.sqrt(dimensions[l - 1])
            self.b[l] = np.random.randn(dimensions[l])/ np.sqrt(dimensions[l])
            self.activations[l] = activations[l - 1]

    def feed_forward(self, x):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """
        # w(x) + b
        a = {}

        # activations: z = f(a)
        z = {0: x}  # First layer has no activations as input, so we consider input itself as the first activation.

        for l in range(1, self.n_layers):
            # current layer = l
            a[l] = z[l - 1] @ self.w[l] + self.b[l]
            z[l] = self.activations[l].forward(a[l])

        return z, a


    def predict(self, x):
        """
        :param x: (array) Containing parameters
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """

        a, _ = self.feed_forward(x)
        return a[self.n_layers - 1]


    def back_prop(self, z, a, y_true):
        """
        The input dicts keys represent the layers of the net.
        a = { 0: x,
              1: f(w1(x) + b1)
              2: f(w2(a2) + b2)
              }
        :param a: (dict) w^T@x + b
        :param z: (dict) f(a)
        :param y_true: (array) One hot encoded truth vector.
        :param loss: Loss class with a static .gradient(y_true, y_pred) method.
        :return:
        """
        # Determine partial derivative and delta for the output layer.
        y_pred = z[self.n_layers - 1]

        #to take gradient with weighted losses
        if self.class_weights is not None:
            delta = self.loss_function.gradient(y_true, y_pred, self.class_weights) * self.activations[self.n_layers - 1].gradient(y_pred)
        else :
            delta = self.loss_function.gradient(y_true, y_pred) * self.activations[self.n_layers - 1].gradient(y_pred)

        dw = np.dot(z[self.n_layers - 2].T, delta)

        update_params = {
            self.n_layers - 1: (dw, delta)
        }

        # Determine partial derivative and delta for the rest of the layers.
        # Each iteration requires the delta from the previous layer, propagating backwards.
        for l in reversed(range(1, self.n_layers - 1)):
            delta = np.dot(delta, self.w[l + 1].T) * self.activations[l].gradient(a[l])
            dw = np.dot(z[l - 1].T, delta)
            update_params[l] = (dw, delta)

        # finally update weights and biases
        for k, v in update_params.items():
            self.update_w_b(k, v[0], v[1])

    def update_w_b(self, index, dw, delta):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        self.w[index] -= self.learning_rate * dw
        self.b[index] -= self.learning_rate * np.mean(delta, 0)

    def fit(self, x, y_true, loss, epochs, batch_size, learning_rate=1e-3, class_weights =None):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :param loss: Loss class (MSE, CrossEntropy etc.)
        :param epochs: (int) Number of epochs.
        :param batch_size: (int)
        :param learning_rate: (flt)
        """
        self.class_weights = class_weights
        self.train_losses = []
        if not x.shape[0] == y_true.shape[0]:
            raise ValueError("Length of x and y arrays don't match")
        # Initiate the loss object with the final activation function
        self.loss_function = loss
        self.learning_rate = learning_rate

        for i in range(epochs):
            # Shuffle the data
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x_ = x[indices]
            y_ = y_true[indices]

            for j in range(x.shape[0] // batch_size):
                k = j * batch_size
                l = (j + 1) * batch_size
                z, a = self.feed_forward(x_[k:l])
                self.back_prop(z, a, y_[k:l])

            if (i + 1) % 10 == 0:
                z, _ = self.feed_forward(x)
                current_loss = self.loss_function.loss(y_true, z[self.n_layers - 1])
                self.train_losses.append(current_loss)
                print("Loss at epoch {}: {}".format(i + 1, current_loss))
