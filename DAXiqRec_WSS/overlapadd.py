import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from numpy import genfromtxt
from numpy import random
import csv
import os
import math
import scipy.signal


f1 = 1000
f2 = 3000
sampleRate = 10000
captureTime = 4
t = np.arange(0, captureTime, 1 /sampleRate)
sig1 = np.sin(t * f1 * 6.2830)
sig2 = np.sin(t * f2 * 6.2830)

noise = .1 * random.rand(captureTime * sampleRate)


sig3 = sig1 + sig2 + noise

plt.plot(sig1[:100])
plt.plot(sig2[:100])
plt.show()

plt.plot(sig3[:100])
plt.show()

fftSig3 = np.fft.fft( sig3[:1024])
fft_samp_abs = np.abs(fftSig3)
normalized = fft_samp_abs / 1024

rms = normalized/1.41421
power = ( (rms **2) / 50 )
db = 10 * np.log10(power )
dbm = db + 30.0
f = np.fft.fftfreq(1024, 1 / sampleRate)

plt.figure(figsize=(15, 10))
plt.xticks(np.arange(-sampleRate/2, sampleRate/2, 2000))
# plt.xticks(f)
# plt.yscale("log")

plt.plot(f, dbm)     #fft_samp_abs)
plt.show()

bpFilter = np.zeros( [1024], dtype=np.complex64)

for inc in range(50, 200, 1):
    bpFilter[inc] = 1 + 0j

filteredSig3 = fftSig3 * bpFilter

fft_samp_abs = np.abs(filteredSig3)
normalized = fft_samp_abs / 1024

rms = normalized/1.41421
power = ( (rms **2) / 50 )
db = 10 * np.log10(power )
dbm = db + 30.0
f = np.fft.fftfreq(1024, 1 / sampleRate)

plt.figure(figsize=(15, 10))
plt.xticks(np.arange(-sampleRate/2, sampleRate/2, 2000))
# plt.xticks(f)
# plt.yscale("log")

plt.plot(f, dbm)     #fft_samp_abs)
plt.show()


dummy = 0
print("done")
