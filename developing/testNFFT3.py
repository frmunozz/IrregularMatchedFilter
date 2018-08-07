from nfft import nfft, ndft, nfft_adjoint, ndft_adjoint
import numpy as np
import matplotlib.pyplot as plt
from nfft.kernels import KERNELS
from nfft.utils import nfft_matrix, fourier_sum, inv_fourier_sum

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
df = 1 / ((max(T) - min(T)) * 5)
min_freq = 0.5 * df
Nf = 1 + int(np.round((max_freq - min_freq) / df))


data_ndft = ndft_adjoint(T, data, Nf)
data_back_dft = ndft(T, (2/len(data)) * data_ndft)

plt.figure(2)
plt.plot(T, data, 'r', label="original")
plt.plot(T, data_back_dft, 'b', label="reconstructed")
plt.title("reconstructed data using NDFT")
plt.legend()
plt.show()

data_nfft = nfft_adjoint(T, data, Nf)
data_back_fft = nfft(T, (2/Nf) * data_nfft)

plt.figure(3)
plt.plot(T, data, 'r', label="original")
plt.plot(T, data_back_fft, 'b', label="reconstructed")
plt.title("reconstructed data using NFFT")
plt.legend()
plt.show()


def new_ndft_adjoint(x: np.ndarray, y: np.ndarray, k: np.ndarray):
    x, f = np.broadcast_arrays(x, y)
    assert x.ndim == 1

    N = len(k)
    assert N % 2 == 0

    fhat = np.dot(f, np.exp(2j * np.pi * k * x[:, None]))

    return fhat


def new_ndft(x, f_hat, k):
    x, f_hat = map(np.asarray, (x, f_hat))
    assert x.ndim == 1
    assert f_hat.ndim == 1

    N = len(k)
    assert N % 2 == 0

    return np.dot(f_hat, np.exp(-2j * np.pi * x * k[:, None]))


max_freq = 30
df = 1 / ((max(T) - min(T)) * 10)
min_freq = 0.5 * df
Nf = 1 + int(np.round((max_freq - min_freq) / df))
k = min_freq + df * np.arange(Nf)


data_nndft = new_ndft_adjoint(T, data, k)
data_back_nndft = new_ndft(T, (2/Nf) * data_nndft, k)

plt.figure(4)
plt.plot(T, data, 'r', label="original")
plt.plot(T, data_back_nndft, 'b', label="reconstructed")
plt.title("reconstructed data using  new NDFT")
plt.legend()
plt.show()





