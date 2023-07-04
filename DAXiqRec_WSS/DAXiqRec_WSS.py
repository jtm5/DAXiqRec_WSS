#########################################################################################################################
#
#   DAXiqRec_WSS.py     4Dec22
#
#   Purpose: record Flex DAX IQ via Windows Sound System (UDP IQ on Flex still questionable given very high values)
#       This is built from SDRExp_6_1.py
#
#   Goal: record varying lengths
#           Output IQ values as .NPY
#           Output MatplotLab graphs of PSD and PS
#           Output array of PSDs as .CSV to be used by wf2.py or similar to show waterfall
#  
#   Intended to debug and make sure of a stable process
#           then to include it in an overall Flex IQ, record, plot and display waterfall as well as DSP of signal
#           
########################################################################################################################     


###########################################################################################################################
# 12 Oct 2022:
#       Quite an improvement. Cleaned up the Overlap Add logic and
#       changed output to a 96000 array of samples.
#       Voice on AM and CHU tones are now pretty intelligble. There is some distortion and an overriding high
#       high-pitched hum.
#       
#       BP filter response seems ok, but with small FFT and filter size it is crude
#
#   ________________________________________________________________________________________
#   NEXT: clean up the OVA logic some more - looks like discontinuities at each 256 boundary
#   _________________________________________________________________________________________
#   
#    
#########################################################################################################################

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
import scipy.signal



fs = 48000  # sample rate
# N = 2048  # number of samples in each FFT
duration = 8    # this sets recording duration in seconds
segSize = 600 #512#128
filterSize = 425 #513#129
FFTN = segSize + filterSize - 1


# Set up capture parameters
devs = sd.default.device
sd.default.device = (47, 17)  # 47 is DAX I/Q 1 and 48??? is DAX 1
devs = sd.default.device

# Do the recording
#myrecording = sd.rec(int(duration*fs), samplerate=fs, channels=2)
#sd.wait()
##myrecording = myrecording / math.sqrt(1000)  # if commented out, no scaling
#np.save("Flex.npy", myrecording)


# OK, assume we have recorded from Flex, input the .NPY file
myrecording = np.load("Flex.npy")

#sd.play(myrecording)
#sd.wait()


myrecLen = len(myrecording)
print("myreclen = ", myrecLen)
sqrt1000 = math.sqrt(1000.0)

complexSamps = np.zeros( len(myrecording), dtype=np.complex64)  

#############################################################
##  
##                  BE CAREFUL - ALL INPUT<<<<<<<<<NOT>>>>>>>>> BEING SCALED
##                                        <<<<<<<< NOT >>>>>>>>
##
##############################################################


for i in range(0, len(myrecording), 1):   
    complexSamps.real[i] = myrecording[i,0] #/ sqrt1000
    complexSamps.imag[i] = myrecording[i,1] #/ sqrt1000


saveFile = "D:\\Visual Studio SOURCE\\DSP Experiments\\complexArray.csv"
np.savetxt(saveFile, complexSamps, delimiter=",")

#pad = np.zeros( (1024), dtype=np.complex64)
sample = np.zeros( (1024), dtype=np.complex64)
sample[0:1024] = complexSamps[0:1024] 

fft_samp = np.fft.fft(sample)
#saveFile = "D:\\Visual Studio SOURCE\\DSP Experiments\\fft_samp.csv"
#np.savetxt(saveFile, fft_samp, delimiter=",")

fft_samp_abs = np.abs(fft_samp)
#saveFile = "D:\\Visual Studio SOURCE\\DSP Experiments\\fft_samp_abs.csv"
#np.savetxt(saveFile, fft_samp_abs, delimiter=",")

normalized = fft_samp_abs / 1024
#saveFile = "D:\\Visual Studio SOURCE\\DSP Experiments\\normalized.csv"
#np.savetxt(saveFile, normalized, delimiter=",")


rms = normalized/1.41421
power = ( (rms **2) / 50 )
db = 10 * np.log10(power )
dbm = db + 30.0
f = np.fft.fftfreq(1024, 1 / fs)

plt.figure(figsize=(15, 10))
plt.xticks(np.arange(-fs/2, fs/2, 2000))
# plt.xticks(f)
# plt.yscale("log")

plt.plot(f, dbm)     #fft_samp_abs)
plt.show()

# two zero padding arrays
zerosSegSize = np.zeros( (segSize), dtype=np.complex64)
zerosSegSizeMinus1 = np.zeros( (segSize - 1), dtype=np.complex64)
bpFilter = np.zeros( (filterSize), dtype=np.complex64)

for m in range(0, filterSize, 1):                  #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                 # NOTE: unless bpfilter is all 1s, the output is < 256 long!!!!
     bpFilter[m] = 1.0 + 0j




# Get the first chunk to be processed loaded intp tempOut
recChunk = complexSamps[0: segSize]

recChunkPad = np.append(recChunk, zerosSegSize)
bpFilterPad = np.append(bpFilter, zerosSegSizeMinus1)
recFFT = np.fft.fft(recChunkPad, FFTN)
outFFT = recFFT * bpFilterPad

audioList = []

# Experiment - try just loading each processed junk into a big array
outRegister = np.zeros(96000, dtype=np.float32)

for inc in range(0, 170, 1):  # max was 765 WHICH WAS TOTALLT STUPID!!!
    recChunk = complexSamps[inc * segSize: (inc + 1) * segSize]
    recChunkPad = np.append(recChunk, zerosSegSize)

    bpFilterPad = np.append(bpFilter, zerosSegSizeMinus1)
    recFFT = np.fft.fft(recChunkPad, FFTN)
    outFFT = recFFT * bpFilterPad

    outAudio = np.fft.ifft(outFFT, FFTN)
    outAudioReal = outAudio.real  * 20     #gain setting

    saveFile = "D:\\Visual Studio SOURCE\\DSP Experiments\\outAudioReal.csv"
    np.savetxt(saveFile, outAudioReal, delimiter=",")
    
    
    # WATCH OUT FOR SETTING A np.array equal to another it is <<<NOT>> a unique array
                                            # and will change anytime the orginal changes!!!  NASTY source of bugs
    
    # OK, now load into the outRegister array
    if (inc == 0):
        for i in range(0, FFTN, 1):
            outRegister[i] += outAudioReal[i]
    else:
        for k in range(0, FFTN, 1):
            outRegister[(inc * 514) + k] += outAudioReal[k]


sd.play(outRegister)
sd.wait()

#sd.play(outRegister)
#sd.wait()

#sd.play(outRegister)
#sd.wait()

plt.title("outRegister")
plt.plot(outRegister[:5000])
plt.show()


f, Pxx_den = scipy.signal.welch(complexSamps[0:1024], fs, nperseg = 2048, scaling="density")
#f, Pxx_den = scipy.signal.welch(complexSamps, fs, nperseg = len(myrecording)/2, scaling="density")

log_Pxx_den = 10 * np.log10( Pxx_den)
#plt.figure()
##plt.semilogy(f, Pxx_den)
#plt.plot(f, log_Pxx_den )
##plt.xlim([0,100])
#plt.xlabel('frequency [Hz]')
#plt.ylabel('Power Spectral Density [V**/hz')
#plt.title('Power Spectral Density (scipy.signal.welch)')
#plt.grid(color='green', linestyle='--')
#plt.show()




f, Pxx_spec = scipy.signal.welch(complexSamps[0:1024], fs, nperseg = 2048, scaling="spectrum")
#f, Pxx_spec = scipy.signal.welch(complexSamps, fs, nperseg = len(myrecording)/2, scaling="spectrum")

log_Pxx_spec = 20 * np.log10( np.sqrt(Pxx_spec) / np.sqrt(50) )
#plt.figure(figsize=(10,20), dpi=80)
#plt.semilogy(f, np.sqrt(Pxx_spec) )# / np.sqrt(50))
##plt.plot(f, log_Pxx_spec )
##plt.xlim([0,100])
#plt.xlabel('frequency [Hz]')
#plt.ylabel('Power Spectrum [V RMS]')
#plt.title('Power Spectrum (scipy.signal.welch)')
#plt.grid(color='green', linestyle='--')
#plt.show()


fig, axs = plt.subplots(2)
fig.suptitle(" scipy.signal.welch")
fig.set_size_inches(15.0,10.0)
axs[0].semilogy(f, np.sqrt(Pxx_spec))
axs[0].set_title("Power Spectrum")
axs[0].set_xlabel('frequency [Hz]')
axs[0].set_ylabel('Power Spectrum [V RMS]')
axs[0].grid(color='red', linestyle='--')
axs[1].plot(f, log_Pxx_den)
axs[1].set_title("Power SpectralDensity")
axs[1].set_xlabel('frequency [Hz]')
axs[1].set_ylabel('Power Spectral Density [V**/hz')
axs[1].grid(color='red', linestyle='--')
#plt.show()


# Now create the array of PSDs
psdArray = np.zeros( ( int(myrecLen/1024) ,1024), dtype=np.float32)
# do a quick PSD just to get the f array to fill first row of PSD array -- for .CSV plotting (& later wf?)
#       NOTE: the PSD output has f bins 0 to +fs/2 in array[0:511]; -fs/2 to 0 is in array[512:1023]
#           Need to rearrange to -f/2 to +fs/2 for waterfall
f, Pxx_den = scipy.signal.welch(complexSamps[0:1024], fs, nperseg = 2048, scaling="density") 
psdArray[0] = f
for i in range(1,int(myrecLen/1024),1):
    f, Pxx_den = scipy.signal.welch(complexSamps[1024 * i  : 1024 * (i+1)], fs, nperseg = 2048, scaling="density")
    psdArray[i] = 10.0 * np.log10(Pxx_den)
    #*****************************************************************************************************************
    #psdArray[i] = np.roll(psdArray[i], 512) # rolls the array to the right  # WHY???????? only if being used by wf4.py
    #*****************************************************************************************************************

saveFile = "D:\\Visual Studio SOURCE\\DSP Experiments\\psd_array.csv"
np.savetxt(saveFile, psdArray, delimiter=",")

np.save("D:\\Visual Studio SOURCE\\DSP Experiments\\psd_array.npy", psdArray)

# EXPERIMENT for animating a PSD display

# FIRST -- try plotting a couple of PSDs from psdArray[]

#plt.figure()
##plt.semilogy(f, Pxx_den)
#pDen=psdArray[200]
#plt.plot(f, pDen )
##plt.xlim([0,100])
#plt.xlabel('frequency [Hz]')
#plt.ylabel('Power Spectral Density [V**/hz')
#plt.title('Power Spectral Density (scipy.signal.welch)')
#plt.grid(color='green', linestyle='--')
#plt.show()


#OK, try a short animation


fig, ax = plt.subplots()

#x = np.arange(0, 2*np.pi, 0.01)
f, Pxx_den = scipy.signal.welch(complexSamps[0:1024], fs, nperseg = 2048, scaling="density") 
x = f
#line, = ax.plot(x, np.sin(x))
line, = ax.plot(x, 10.0 * np.log10(Pxx_den))

def animate(i):
 
    line.set_ydata(psdArray[i])  # update the data.
    i += 1
    if (i > 374):
        ani.event_source.stop()
    return line,


ani = animation.FuncAnimation(
    fig, animate, interval=50, blit=True, save_count=50)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)


#plt.show()


