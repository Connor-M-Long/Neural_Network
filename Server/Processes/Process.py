import random
import numpy as np
import pandas as pd
from Client.UI import interface
from Server.Logic import Math
from Server.Entities import weightsBiases

class Data:
    def __init__(self):
        pass

    def get_data(self):
        data = pd.read_csv('train_info/train.csv')

        data = np.array(data)
        rows, columns = data.shape
        np.random.shuffle(data)

        data_dev = data[0:1000].T  # checks each column
        Y_dev = data_dev[0]  # this is  the name of the image
        X_dev = data_dev[1:columns]  # the pixels in the image
        X_dev = X_dev / 255

        data_train = data[1000:rows].T
        Y_train = data_train[0]  # this is  the name of the image
        X_train = data_train[1:columns]  # the pixels in the image
        X_train = X_train / 255
        _, m_train = X_train.shape

        NN = Math.NeuralNeural()

        W1, b1, W2, b2 = NN.gradient_descent(X_train, Y_train, 0.10, 500)  # trains the neural network

        #values = weightsBiases.DTO(W1, W2, b1, b2, X_dev, Y_dev)

        #return values

        self.get_prediction(W1, b1, W2, b2, X_dev, Y_dev)

    def get_prediction(self, W1, b1, W2, b2, X_dev, Y_dev):

        NN = Math.NeuralNeural()
        pred, lbl, ci = NN.test_prediction(random.randint(0, 9), W1, b1, W2, b2, X_dev, Y_dev)  # creates a prediction

        interface.display(pred, lbl, ci, W1, b1, W2, b2, X_dev, Y_dev)