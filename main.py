from Server.Processes import Process

data = Process.Data()
pred, lbl, img = data.get_prediction()

print("Prediction: " + str(pred))
print("Value: " + str(lbl))
img.show()