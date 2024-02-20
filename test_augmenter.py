from libemg.data_augmenter import DataAugmenter
import numpy as np 
import matplotlib.pyplot as plt

da = DataAugmenter()

data = np.loadtxt("R_0_C_0.csv", delimiter=',')

# data2 = da.aug60HZ(data, 0, 200)
data3 = da.augMA(data, 0, 0.3, 200)
# data2 = da.augMIXUP(data)

fig, axs = plt.subplots(2)
axs[0].plot(data)
axs[1].plot(data3)
plt.show()
