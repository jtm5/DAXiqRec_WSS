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
#t = np.arange(0, captureTime, 1 /sampleRate)
#sig1 = np.sin(t * f1 * 6.2830)
#sig2 = np.sin(t * f2 * 6.2830)



def generateTone(fs, toneFreq, numSamples, amplitude):
    #FROM: https://github.com/jgibbard/iqtool/blob/master/iqgen.py

    #Copyright (c) 2017 James Gibbard

    #Generates an IQ data file based on provided parameters
    #Tested with python 2.7 and 3.6
    #Requires numpy and matplotlib
    
    step = (float(toneFreq) / float(fs)) * 2.0 * np.pi
    
    phaseArray = np.array(range(0,numSamples)) * step
    
    #Euler's Formular: e^(j*theta) = cos(theta) + j * sin(theta)
    #For a complex sinusoidal theta = 2*pi*f*t where each time step is 1/fs    
    wave = np.exp(1.0j * phaseArray) * amplitude
    
    return wave

if __name__ == '__main__':
    sig1 = generateTone(sampleRate, 700, captureTime * sampleRate, 2)
    sig2 = generateTone(sampleRate, 800, captureTime * sampleRate, 1)
    sig3 =sig1 #+sig2

noise = .8 * random.rand(captureTime * sampleRate)
sig3 = sig3 + sig2
bpFilter = np.zeros( [1024], dtype=np.complex64)


## 8jul23 test of designing a windowed-sinc LP filter - see window_sinc.py part of this solution
#TCalculate a windowed-sinc lp filter     8jul23
#       this is from https://tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter

# NOTE: below, I picked b to give me a 425 tap filter to match what had done before in original overlapadd.py
fc = 0.2  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
b = 0.00945  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
N = int(np.ceil((4 / b)))
if not N % 2: N += 1  # Make sure that N is odd.
n = np.arange(N)
 
# Compute sinc filter.
h = np.sinc(2 * fc * (n - (N - 1) / 2))
 
# Compute Blackman window.
w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
    0.08 * np.cos(4 * np.pi * n / (N - 1))
 
# Multiply sinc filter by window.
h = h * w
 
# Normalize to get unity gain.
h = h / np.sum(h)
plt.plot(h)
plt.show()
h_FFT = np.fft.fft(h,1024)
#plt.semilogy(np.absolute(h_FFT))
#plt.show()

#filteredSig3 = fftSig3 * h_FFT
#plt.plot(filteredSig3)
#plt.show()


###############################################################################################################
###############################################################################################################
# Overlap add code experiment starts here:

# Trying defining signal chunks as 600 samples
#   Filter size will be 425 bins
#       So, need to pad sig chunk with 424 zero
#       And, pad filter with 599 zeros

chunkSize = 600
filterSize = 425
chunk = np.zeros( [1024], dtype=np.complex64)
bpFilter = np.zeros( [1024], dtype=np.complex64)


for i in range(0, 425,1):
    bpFilter[i] = h[i]

bpFilterFFT = np.fft.fft(bpFilter)
plt.semilogy( np.absolute(bpFilterFFT))
plt.show()

# array to hold the iFFT outputs
outPut = np.zeros( [captureTime * sampleRate], dtype=np.complex64)

# First, try processing 5 chunks to see how it works
for i in range(0, 8, 1):
    chunk[:600] = sig3[ i * 600 : (i * 600) + 600]
    chunkFFt = np.fft.fft(chunk) #/ 1024
    #plt.title("chunk")
    #plt.plot(np.absolute(chunkFFt))
    #plt.show()
    filteredChunk = chunkFFt * bpFilterFFT # h_FFT
    #plt.title("filtered chunk")
    #plt.plot(np.absolute(filteredChunk))
    #plt.show()
    chunkOut = np.fft.ifft(filteredChunk)
    #plt.title("chunkOut")
    #plt.plot( np.real(chunkOut) )
    #plt.show()

    if ( i == 0):
        outPut[i * 1024 : (i * 1024) + 1024] = chunkOut[:1024]
    else:
        outPut[i * 600 : (i * 600) + 1024] = outPut[i * 600 : (i * 600) + 1024] + chunkOut[:1024]

#plt.plot(np.real(outPut[:5000]))
#plt.show()

#plt.plot(np.imag(outPut[:5000]))
#plt.show()

fig, axs = plt.subplots(2)
fig.suptitle(" output")
fig.set_size_inches(15.0,10.0)
axs[0].plot(np.real(outPut[0:5000]))
axs[0].set_title("Real")
#axs[0].set_xlabel('frequency [Hz]')
#axs[0].set_ylabel('Power Spectrum [V RMS]')
axs[0].grid(color='red', linestyle='--')
axs[1].plot(np.imag(outPut[0:5000]))
axs[1].set_title("Imaginary")
#axs[1].set_xlabel('frequency [Hz]')
#axs[1].set_ylabel('Power Spectral Density [V**/hz')
axs[1].grid(color='red', linestyle='--')
plt.show()

