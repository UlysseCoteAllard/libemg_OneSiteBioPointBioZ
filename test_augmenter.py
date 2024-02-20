from libemg.data_augmenter import DataAugmenter
import numpy as np 
import matplotlib.pyplot as plt

da = DataAugmenter()

data = np.loadtxt("R_0_C_0.csv", delimiter=',')

data2 = da.augMIXUP(data)

fig, axs = plt.subplots(2)
axs[0].plot(data)
axs[1].plot(data2)
plt.show()
