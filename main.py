from Timer import Timing
from Processes import Process

t = Timing.Timer
t.start()

data = Process.Data
data.get_data()

t.stop()