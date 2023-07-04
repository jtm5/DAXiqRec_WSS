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

#plt.plot(sig1[:100])
#plt.plot(sig2[:100])
#plt.show()

plt.plot(sig3[:100])
#plt.show()

fftSig3 = np.fft.fft( sig3[:1024])
fft_samp_abs = np.abs(fftSig3)
normalized = fft_samp_abs / 1024

rms = normalized/1.41421
power = ( (rms **2) / 50 )
db = 10 * np.log10(power )
dbm = db + 30.0
f = np.fft.fftfreq(1024, 1 / sampleRate)

#plt.figure(figsize=(15, 10))
#plt.xticks(np.arange(-sampleRate/2, sampleRate/2, 2000))
# plt.xticks(f)
# plt.yscale("log")

#plt.plot(f, dbm)     #fft_samp_abs)
#plt.show()

bpFilter = np.zeros( [1024], dtype=np.complex64)

# make the filter about 60db attenuation
#  Note:  this filter is pur brick wall - need to improve
#       by doing a real filter design and/or windowing

for inc in range(0, 1024, 1):
    bpFilter[inc] = .001 + 0j

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

np.savetxt("fft.csv", dbm, delimiter=',')

#plt.figure(figsize=(15, 10))
#plt.xticks(np.arange(-sampleRate/2, sampleRate/2, 2000))
## plt.xticks(f)
# plt.yscale("log")

plt.plot(f, dbm)     #fft_samp_abs)
plt.show()

# Overlap add code experiment starts here:

# Trying defining signal chunks as 600 samples
#   Filter size will be 425 bins
#       So, need to pad sig chunk with 424 zero
#       And, pad filter with 599 zeros

chunkSize = 600
filterSize = 425
chunk = np.zeros( [1024], dtype=np.complex64)
bpFilter = np.zeros( [1024], dtype=np.complex64)

# make the filter about 60db attenuation
#  Note:  this filter is pur brick wall - need to improve
#       by doing a real filter design and/or windowing



for inc in range(0, 425, 1):    # making full band pass for now
    bpFilter[inc] = 1 + 0j

#for inc in range(50, 200, 1):    
#    bpFilter[inc] = 1 + 0j

bpFilterFFT = np.fft.fft(bpFilter) / 1024

# array to hold the iFFT outputs
outPut = np.zeros( [40000], dtype=np.complex64)

# First, try processing 5 chunks to see how it works
for i in range(0, 5, 1):
    chunk[:600] = sig3[ i * 600 : (i * 600) + 600]
    chunkFFt = np.fft.fft(chunk) / 1024
    plt.plot(np.absolute(chunkFFt))
    plt.show()
    filteredChunk = chunkFFt * bpFilterFFT
    plt.plot(np.absolute(filteredChunk))
    plt.show()
    chunkOut = np.fft.ifft(filteredChunk)

    outPut[i * 600 : (i * 600) +1024] = chunkOut[:1024]

plt.plot(np.real(outPut[:5000]))
plt.show()


dummy = 0
print("done")
