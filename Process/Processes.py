import numpy as np

class NeuralNeural:
    def __init__(self):
        self.weight_1 = np.random.rand(10, 784) - 0.5
        self.bias_1 = np.random.rand(10, 1) - 0.5
        self.weight_2 = np.random.rand(10, 10) - 0.5
        self.bias_2 = np.random.rand(10, 1) - 0.5

    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def ReLU(self, z):
        return np.maximum(z, 0)

    def ReLU_deriv(self, z):
        return z > 0

    def change_val(self, Y):   # an array of targets
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def forward_prop(self, W1, b1, W2, b2, X):
        Z1 = W1.dot(X) + b1
        A1 = self.ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def backward_prop(self, Z1, A1, Z2, A2, W1, W2, X, Y):
        m = Y.size   # length of the input
        one_hot_Y = self.change_val(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)
        dZ1 = W2.T.dot(dZ2) * self.ReLU_deriv(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)
        return dW1, db1, dW2, db2

    def update_params(self, W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        return W1, b1, W2, b2

    def gradient_descent(self, X, Y, alpha, iterations):
        W1, b1, W2, b2 = self.weight_1, self.bias_1, self.weight_2, self.bias_2

        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_prop(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
            W1, b1, W2, b2 = self.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = self.get_predictions(A2)
                print(self.get_accuracy(predictions, Y), "%")

        return W1, b1, W2, b2

    def get_predictions(self, A2):
        return np.argmax(A2, 0)

    def get_accuracy(self, predictions, Y):
        print(predictions, Y)
        return round((np.sum(predictions == Y) / Y.size) * 100, 2)

    def make_predictions(self, X, W1, b1, W2, b2):
        _, _, _, A2 = self.forward_prop(W1, b1, W2, b2, X)
        predictions = self.get_predictions(A2)
        for x in predictions:
            return x

    def test_prediction(self, index, W1, b1, W2, b2, X, Y):
        current_image = X[:, index, None]
        prediction = self.make_predictions(X[:, index, None], W1, b1, W2, b2)   # picks the array in matrix X_train
        label = Y[index]

        return prediction, label, current_image
