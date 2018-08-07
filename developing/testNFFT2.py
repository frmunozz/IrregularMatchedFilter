"""
check some doubt in the behavior of NFFT

"""
from nfft import nfft_adjoint, ndft_adjoint
import nfft
from nfft.kernels import KERNELS
from nfft.utils import nfft_matrix, fourier_sum, inv_fourier_sum

import numpy as np
import matplotlib.pyplot as plt

# first: can i use a specified range of frequency? for this need to modified
# the nfft implementation

# 3 parts separated in time, one with slight irregularities in time sampling
# another with change of spacing and the last one with big outlier in spacing
N = 120
T = np.zeros(N)
dt_implicit = 1 / N
t0 = np.linspace(0, int(N/3)-1, int(N/3))
np.random.seed(1)
e = np.random.normal(0, dt_implicit * 0.5, N//3)
T[0:N//3] = t0 * dt_implicit + e
shift = 30 * dt_implicit

np.random.seed(2)
t0 = np.linspace(int(N/3), int(N*1/2)-1, int(N/6))
e = np.random.normal(0, dt_implicit * 0.5, N//6)
T[N//3:N//2] = shift + t0 * dt_implicit / 2 + e

np.random.seed(3)
t0 = np.linspace(int(N/2), int(N*2/3)-1, int(N/6))
e = np.random.normal(0, dt_implicit * 0.5, N//6)
T[N//2:2*N//3] = t0 * 2 * dt_implicit + e

np.random.seed(4)
t0 = np.linspace(2*N//3, N-1, N - 2*N//3)
e = np.random.normal(0, dt_implicit * 0.5, N - 2*N//3)
T[2*N//3:N] = 2 * shift + t0 * dt_implicit / 2 + e
T.sort()

#  signal is sinusoidal again with same frequency
freq_of_sin = 10
s = np.sin(freq_of_sin * 2 * np.pi * T)
#  apply noise
np.random.seed(1)
noise = np.random.normal(0, 0.3, N)
data = s + noise
plt.figure(0)
plt.plot(T, data, alpha=0.5)
plt.plot(T, s, "k.-")
plt.show()

max_freq = 30
NN = int(round(max(T) * 5)) * max_freq * 2
df = 1 / (max(T) - min(T))
k = (-(NN//2) + np.arange(NN)) * df
# N = np.arange(N)
data_ndft = ndft_adjoint(T, data, NN)
data_nfft = nfft_adjoint(T, data, NN)
plt.figure(1)
plt.plot(k, data_ndft, 'r')
plt.plot(k, data_nfft, 'b')
plt.show()


def new_nfft_adjoint(x: np.ndarray, y: np.ndarray, k: np.ndarray, sigma=3,
                     tol=1E-8, m=None, kernel='gaussian',
                     use_fft=True, truncated=True):
    # Validate inputs
    x, y = np.broadcast_arrays(x, y)
    assert x.ndim == 1

    sigma = int(sigma)
    assert sigma >= 2

    N = len(k)
    assert N % 2 == 0

    n = N * sigma

    kernel = KERNELS.get(kernel, kernel)

    if m is None:
        m = kernel.estimate_m(tol, N, sigma)

    m = int(m)
    assert m <= n // 2

    # Compute the adjoint NFFT
    mat = nfft_matrix(x, n, m, sigma, kernel, truncated=truncated)
    g = mat.T.dot(y)
    ghat = inv_fourier_sum(g, N, n, use_fft=use_fft)

    fhat = ghat / kernel.phi_hat(k, n, m, sigma) / n

    return fhat


def new_ndft_adjoint(x: np.ndarray, y: np.ndarray, k: np.ndarray):
    x, f = np.broadcast_arrays(x, y)
    assert x.ndim == 1

    N = len(k)
    assert N % 2 == 0

    fhat = np.dot(f, np.exp(2j * np.pi * k * x[:, None]))

    return fhat


df = 1 / ((max(T) - min(T)) * 5)
min_freq = 0.5 * df
Nf = 1 + int(np.round((max_freq - min_freq) / df))
k = min_freq + df * np.arange(Nf)

plt.figure(2)
new_data_nfft = new_nfft_adjoint(T, data, k)
plt.plot(k, np.abs(new_data_nfft))
plt.show()

plt.figure(3)
new_data_nfft = new_ndft_adjoint(T, data, k)
plt.plot(k, np.abs(new_data_nfft))
plt.show()





