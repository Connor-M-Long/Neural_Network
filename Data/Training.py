import PySimpleGUI as sg
from matplotlib import pyplot as plt
import random
import numpy as np
import pandas as pd
from Process import Processes

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

        NN = Processes.NeuralNeural()

        W1, b1, W2, b2 = NN.gradient_descent(X_train, Y_train, 0.10, 500)  # trains the neural network

        self.get_prediction(W1, b1, W2, b2, X_dev, Y_dev)

    def get_prediction(self, W1, b1, W2, b2, X_dev, Y_dev):

        NN = Processes.NeuralNeural()
        pred, lbl, ci = NN.test_prediction(random.randint(0, 9), W1, b1, W2, b2, X_dev, Y_dev)  # creates a prediction
        self.display(pred, lbl, ci, W1, b1, W2, b2, X_dev, Y_dev)

    def display(self, prediction, label, image, W1, b1, W2, b2, X_dev, Y_dev):

        current_image = image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.savefig("Num.png")

        col1 = [[sg.Text("Image Recognition", font="Monospace, 22")],
                [sg.Text("Prediction: ", font="Monospace")] + [sg.Text(prediction, font="Monospace", key="pred")],
                [sg. Text("Value: ", font="Monospace")] + [sg.Text(label, font="Monospace", key="lbl")],
                [sg.Text()], [sg.Text()], [sg.Text()], [sg.Text()],
                [sg.Button("Make Predictions", font="Monospace")] +
                [sg.Button("OK", font="Monospace")]]

        col2 = [[sg.Image(r'num.png', key="image")]]

        layout = [[sg.Column(col1), sg.Column(col2)]]

        window = sg.Window("Neural Network", layout, size=(580, 300), resizable=True, finalize=True)
        window.maximize()

        while True:
            event, values = window.read()

            if event == "Make Predictions":
                NN = Processes.NeuralNeural()
                pred, lbl, ci = NN.test_prediction(random.randint(0, 9), W1, b1, W2, b2, X_dev, Y_dev)
                window["pred"].update(pred)
                window["lbl"].update(lbl)

                current_image = ci.reshape((28, 28)) * 255
                plt.gray()
                plt.imshow(current_image, interpolation='nearest')
                plt.savefig("Num.png")

                window["image"].update("Num.png")

            if event == "OK" or event == sg.WIN_CLOSED:
                break

        window.close()