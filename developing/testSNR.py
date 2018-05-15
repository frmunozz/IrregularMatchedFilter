from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from scipy import signal
"""
trying to obtain the SNR by duplicating the code from LIGO tutorial.
test for signal of only noise, this should show the normalization onf SNR.
"""
# sinusoidal signal
N = 4096  # should be even for simplicity
dt = 1 / 4096
t = np.linspace(0, N * dt, N)
s = np.sin(10 * 2 * np.pi * t)

# set data as only noise
np.random.seed(2)
data = np.random.normal(0, 1, N)

fs = int(1 / dt)  # sampling rate
NFFT = int(4 * fs)
psd_window = np.blackman(NFFT)
dwindow = signal.tukey(s.size, alpha=1./8)
NOVL = int(NFFT / 2)
datafreq = np.fft.fftfreq(s.size)*fs
df = np.abs(datafreq[1] - datafreq[0])
template_fft = np.fft.fft(s*dwindow) / fs

data_psd, freqs = mlab.psd(data)

# Take the Fourier Transform (FFT) of the data and the template (with dwindow)
data_fft = np.fft.fft(data*dwindow) / fs

# -- Interpolate to get the PSD values at the needed frequencies
power_vec = np.interp(np.abs(datafreq), freqs, data_psd)
plt.figure()
plt.plot(freqs, data_psd, "g")
# plt.plot(np.abs(datafreq), power_vec, "r")
plt.show()
# -- Calculate the matched filter output in the time domain:
# Multiply the Fourier Space template and data, and divide by the noise power in each frequency bin.
# Taking the Inverse Fourier Transform (IFFT) of the filter output puts it back in the time domain,
# so the result will be plotted as a function of time off-set between the template and the data:
optimal = data_fft * template_fft.conjugate() / power_vec
optimal_time = 2*np.fft.ifft(optimal)*fs

# -- Normalize the matched filter output:
# Normalize the matched filter output so that we expect a value of 1 at times of just noise.
# Then, the peak of the matched filter output will tell us the signal-to-noise ratio (SNR) of the signal.
sigmasq = 1*(template_fft * template_fft.conjugate() / power_vec).sum() * df
sigma = np.sqrt(np.abs(sigmasq))
SNR_complex = optimal_time/sigma

plt.plot(t, SNR_complex.real)
print(np.mean(SNR_complex), np.mean(np.abs(SNR_complex)), np.mean(SNR_complex.real))
plt.show()
