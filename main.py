from Server.Processes import Process, Timer

t = Timer.Timer
t.start()

data = Process.Data
data.get_data()

t.stop()