#TCalculate a windowed-sinc lp filter     8jul23
#       this is from https://tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter


#from __future__ import division
 
import numpy as np
import matplotlib.pyplot as plt
 
fc = 0.2  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
b = 0.0094  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
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
plt.semilogy(np.absolute(h_FFT))
plt.show()

fig, axs = plt.subplots(2)
fig.suptitle(" h_FFT")
fig.set_size_inches(15.0,10.0)
axs[0].plot(np.real(h_FFT))
axs[0].set_title("Real")
#axs[0].set_xlabel('frequency [Hz]')
#axs[0].set_ylabel('Power Spectrum [V RMS]')
axs[0].grid(color='red', linestyle='--')
axs[1].plot(np.imag(h_FFT))
axs[1].set_title("Imaginary")
#axs[1].set_xlabel('frequency [Hz]')
#axs[1].set_ylabel('Power Spectral Density [V**/hz')
axs[1].grid(color='red', linestyle='--')
plt.show()




dummy = 0