#experiment to learn numpy convolve function for sig proc
#27dec21, jtm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from numpy import genfromtxt
from numpy import random
import csv
import os
f1 = 4
f2 = 15
t = np.arange(0,1,0.01)
sig1 = np.sin(t * f1 * 6.2830)
sig2 = np.sin(t * f2 * 6.2830)

noise = random.rand(100)


sig3 = sig1 + sig2 + noise #* 5

plt.plot(sig1)
plt.plot(sig2)
#plt.show()

plt.plot(sig3)
#plt.show()

output=np.convolve(sig1, sig3, 'same')
plt.plot(output)
plt.show()


print("done")

