import nfft
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
from core.SSFunction import SS
import pdb
"""
Testing implementation of NFFT of jake vanderplas
"""


# sinusoidal signal
N = 600  # should be even for simplicity
dt = 1 / 800
t = np.linspace(0, N * dt, N)
nyq = 1 / (2 * dt)
s = np.sin(10 * 2 * np.pi * t)

# add some noise
np.random.seed(1)
noise = np.random.normal(0, 0.3, N)
data = s + noise
plt.figure(1)
plt.plot(t, data, alpha=0.5)
plt.plot(t, s, "k.-")
plt.show()

# try the fft from numpy
d_fft = fft(data)
# try the nfft from NFFT
d_nfft = nfft.nfft(t, data)
# try the ndft from NFFT
d_ndft = nfft.ndft(t, data)

xf = np.linspace(0.0, nyq, int(N / 2))
plt.figure(2)
plt.plot(xf, (2 / N) * np.abs(d_fft[:N//2]), 'r--', label="fft")
plt.plot(xf, (2 / N) * np.abs(d_nfft[:N//2]), 'g--', label="nfft")
plt.plot(xf, (2 / N) * np.abs(d_ndft[:N//2]), 'b-', label="ndft", alpha=0.5)
b = np.allclose(d_nfft[:N//2], d_ndft[:N//2])
plt.title("comparison between ft's from regular sampling")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Fourier Transform")
plt.text(190, 0.98, "np.allclose(nfft, ndft) = {}".format(b))
plt.legend()
plt.show()

"""
now test for non uniform data, here we are going to interpolate the signal just to
compare to fft, for that we will make a data almost uniform
"""

N = 600
dt_implicit = 1 / 800
t = np.linspace(0, N-1, N)
np.random.seed(1)
e = np.random.normal(0, dt * 0.5, N)
t = t * dt_implicit + e
t.sort()

#  signal is sinusoidal again with same frequency
s = np.sin(10 * 2 * np.pi * t)
#  apply same noise
data = s + noise
plt.figure(3)
plt.plot(t, data, alpha=0.5)
plt.plot(t, s, "k.-")
plt.show()

# try the fft from numpy
regular_t = np.linspace(min(t), max(t), len(data))
regular_data = np.interp(regular_t, t, data)
d_fft = fft(regular_data)
# try the nfft from NFFT
d_nfft = nfft.nfft(t, data)
# try the ndft from NFFT
d_ndft = nfft.ndft(t, data)

# get the nyq freq (this is not necessary)
# nyq = SS().nyq(t, show_plot=True)
# plt.show()
# xf = np.linspace(1, int(N / 2), int(N / 2))

plt.figure(4)
plt.plot((2 / N) * np.abs(d_fft[:N//2]), 'r--', label="fft")
plt.plot((2 / N) * np.abs(d_nfft[:N//2]), 'g--', label="nfft")
plt.plot((2 / N) * np.abs(d_ndft[:N//2]), 'b-', label="ndft", alpha=0.5)
b = np.allclose(d_nfft[:N//2], d_ndft[:N//2])
plt.title("comparison between ft's from slight non-regular sampling")
plt.xlabel("sequence number (dimensionless)")
plt.ylabel("Fourier Transform")
plt.text(120, 0.98, "np.allclose(nfft, ndft) = {}".format(b))
plt.legend()
plt.show()

"""
for last, test using highly irregular samplig to check that NFFT works
better than FFT interpolating.
"""

# 3 parts separated in time, one with slight irregularities in time sampling
# another with change of spacing and the last one with big outlier in spacing
N = 600
T = np.zeros(N)
dt_implicit = 1 / 800
t0 = np.linspace(0, int(N/3)-1, int(N/3))
np.random.seed(1)
e = np.random.normal(0, dt * 0.5, N//3)
T[0:N//3] = t0 * dt_implicit + e
shift = 30 * dt_implicit

np.random.seed(2)
t0 = np.linspace(int(N/3), int(N*1/2)-1, int(N/6))
e = np.random.normal(0, dt * 0.5, N//6)
T[N//3:N//2] = shift + t0 * dt_implicit / 2 + e

np.random.seed(3)
t0 = np.linspace(int(N/2), int(N*2/3)-1, int(N/6))
e = np.random.normal(0, dt * 0.5, N//6)
T[N//2:2*N//3] = t0 * 2 * dt_implicit + e

np.random.seed(4)
t0 = np.linspace(int(2*N/3), N-1, int(N/3))
e = np.random.normal(0, dt * 0.5, N//3)
T[2*N//3:N] = 2 * shift + t0 * dt_implicit / 2 + e
T.sort()

#  signal is sinusoidal again with same frequency
s = np.sin(10 * 2 * np.pi * t)
#  apply same noise
data = s + noise
plt.figure(5)
plt.plot(T, data, alpha=0.5)
plt.plot(T, s, "k.-")
plt.show()

# try the fft from numpy
regular_t = np.linspace(min(T), max(T), len(data))
regular_data = np.interp(regular_t, T, data)
d_fft = fft(regular_data)
# try the nfft from NFFT
d_nfft = nfft.nfft(T, data)
# try the ndft from NFFT
d_ndft = nfft.ndft(T, data)

plt.figure(5)
plt.plot((2 / N) * np.abs(d_fft[:N//2]), 'r--', label="fft")
plt.plot((2 / N) * np.abs(d_nfft[:N//2]), 'g--', label="nfft")
plt.plot((2 / N) * np.abs(d_ndft[:N//2]), 'b-', label="ndft", alpha=0.5)
b = np.allclose(d_nfft[:N//2], d_ndft[:N//2])
plt.title("comparison between ft's from highly non-regular sampling")
plt.xlabel("sequence number (dimensionless)")
plt.ylabel("Fourier Transform")
plt.text(120, 0.98, "np.allclose(nfft, ndft) = {}".format(b))
plt.legend()
plt.show()
