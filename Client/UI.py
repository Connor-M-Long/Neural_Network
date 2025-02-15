import PySimpleGUI as sg
from matplotlib import pyplot as plt
from Server.Logic import Math
import random

class interface:
    def __init__(self):
        pass

    def display(self, prediction, label, image, W1, b1, W2, b2, X_dev, Y_dev):
        current_image = image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.savefig("Num.png")

        col1 = [[sg.Text("Image Recognition", font="Monospace, 22")],
                [sg.Text("Prediction: ", font="Monospace")] + [sg.Text(prediction, font="Monospace", key="pred")],
                [sg.Text("Value: ", font="Monospace")] + [sg.Text(label, font="Monospace", key="lbl")],
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
                NN = Math.NeuralNeural()
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