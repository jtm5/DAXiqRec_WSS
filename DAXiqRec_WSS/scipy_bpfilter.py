from cmath import sqrt
import signal
import time
from tkinter.font import NORMAL
#from types import DynamicClassAttribute
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.animation as animation
import sounddevice as sd
import numpy as np  # Make sure NumPy is loaded before it is used in the callback
assert np  # avoid "imported but unused" message (W0611)
import math
from scipy import signal



fs = 48000  # sample rate
# N = 2048  # number of samples in each FFT
duration = 8    # this sets recording duration in seconds
segSize = 600 #512#128
filterSize = 425 #513#129
FFTN = segSize + filterSize - 1

numtaps = 425

f1, f2 = 0.020833, 0.0625
result = signal.firwin(numtaps, [f1, f2], width = 0.041, window='blackman',pass_zero=False) #, fs=48000)

plt.plot(result)
plt.show()

fft_result = np.fft.fft(result,2048)
plt.semilogy(np.absolute(fft_result))
plt.show()



#array([ 0.06301614,  0.88770441,  0.06301614])
dummy = 0
