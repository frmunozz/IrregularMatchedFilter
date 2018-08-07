from numpy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from scipy.signal import lombscargle
import scipy.signal as signal
"""
Trying to obtain the SNR by following the paper and the implementation pf PyCBC
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
df = 1 / (N * dt)
freqs = fftfreq(N)
freqs_lomb = np.delete(np.abs(freqs), 0)

psd, freqs_psd = mlab.psd(data)
pgram = lombscargle(t, data, freqs_lomb, normalize=True)
plt.figure(1)
plt.plot(freqs_lomb, pgram, "b")
plt.show()
# print(np.mean(psd), np.mean(pgram))
# dwindow = signal.tukey(s.size, alpha=1./8)
# # pgram = np.mean(pgram)
# fft_d = np.delete(fft(dwindow * data), 0) # remving the value corresponding to 0 frequency
# fft_t = np.delete(fft(dwindow * s), 0)
#
# norm_sigma = 4 * df
# h_norm = (fft_t * fft_t.conjugate() / pgram).sum()
# norm_corr = 4 * df / np.sqrt(h_norm.real * norm_sigma)
# corr = fft_d * fft_t.conjugate() / pgram
# snr = ifft(corr) * norm_corr * dt * (len(fft_d) - 1)
# snr = np.roll(snr, len(snr) // 2)
# plt.figure(2)
# plt.plot(np.abs(snr))
# plt.show()

