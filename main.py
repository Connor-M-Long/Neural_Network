from Timer import Timing
from Data import Training


t = Timing
t.Timer.start()

data = Training
data.Data.get_data()

t.Timer.stop()

