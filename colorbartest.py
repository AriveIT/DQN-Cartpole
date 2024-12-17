import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(low=0, high=10, size=(25,))
y = np.random.uniform(low=0, high=10, size=(25,))
z = np.random.uniform(low=0, high=10, size=(25,))

fig, axs = plt.subplots(2,1)
axs[0].plot(x)
p=axs[1].scatter(x, y, c=z)
fig.colorbar(p)
plt.show()
