import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from Server.Logic import Math

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

        storedData = {
            "w1": W1.tolist(),
            "w2": W2.tolist(),
            "b1": b1.tolist(),
            "b2": b2.tolist(),
            "x": X_dev.tolist(),
            "y": Y_dev.tolist()
        }

        json_storedData = json.dumps(storedData)

        with open('./Configuration/wbConfig.json', mode="w", encoding="utf-8") as file:
            json.dump(json_storedData, file, indent=4)

    def get_prediction(self):

        with open('./Configuration/wbConfig.json', mode="r") as file:
            config = json.load(file)

        if isinstance(config, str):
            config = json.loads(config)

        W1, W2, b1, b2, X_dev, Y_dev = config["w1"], config["w2"], config["b1"], config["b2"], config["x"], config["y"]

        X_dev = np.array(X_dev)
        Y_dev = np.array(Y_dev)
        W1 = np.array(W1)
        W2 = np.array(W2)
        b1 = np.array(b1)
        b2 = np.array(b2)

        NN = Math.NeuralNeural()
        pred, lbl, ci = NN.test_prediction(random.randint(0, 9), W1, b1, W2, b2, X_dev, Y_dev)  # creates a prediction

        current_image = ci.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.savefig("Configuration/Num.png")

        img = Image.open("Configuration/Num.png")

        return pred, lbl, img