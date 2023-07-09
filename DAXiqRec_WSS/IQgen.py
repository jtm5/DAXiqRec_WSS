FROM: https://github.com/jgibbard/iqtool/blob/master/iqgen.py

#Copyright (c) 2017 James Gibbard

#Generates an IQ data file based on provided parameters
#Tested with python 2.7 and 3.6
#Requires numpy and matplotlib

import argparse
from sys import byteorder
import numpy as np
import matplotlib.pyplot as plt



def generateTone(fs, toneFreq, numSamples, amplitude):
    #Generates a sinusoidal signal with the specified
    #frequency and amplitude
    
    step = (float(toneFreq) / float(fs)) * 2.0 * np.pi
    
    phaseArray = np.array(range(0,numSamples)) * step
    
    #Euler's Formular: e^(j*theta) = cos(theta) + j * sin(theta)
    #For a complex sinusoidal theta = 2*pi*f*t where each time step is 1/fs    
    wave = np.exp(1.0j * phaseArray) * amplitude
    
    return wave

if __name__ == '__main__':
    sig1 = generateTone(10000, 2000, 40000, 2)
    plt.plot(np.real(sig1[0:100]))
    plt.show()
    plt.plot(np.imag(sig1[0:100]))
    plt.show()
    sig2 = generateTone(10000, 3000, 40000, 1)
    sig3 =sig1 +sig2


    dummy = 0
