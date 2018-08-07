from astropy.stats import LombScargle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nfft
import scipy.signal as signal
plt.style.use('seaborn-paper')


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
data = noise
plt.figure(0)
plt.plot(T, data, alpha=0.5)
plt.plot(T, s, "k.-")
plt.show()

frequency, power = LombScargle(T, data).autopower()
frequency = frequency[:len(frequency)-1]
power = power[:len(power)-1]
plt.figure(1)
plt.plot(frequency, power)
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


dwindow = signal.tukey(s.size, alpha=1./8)
data_nfft = new_ndft_adjoint(T, dwindow * data, frequency)
template_nfft = new_ndft_adjoint(T, dwindow * s, frequency)

plt.figure(6)
plt.plot(frequency, np.abs(data_nfft), 'r--', label="data")
plt.plot(frequency, np.abs(template_nfft), 'b', alpha=0.5, label="template")
plt.axvline(freq_of_sin, color='k')
plt.legend()
plt.show()

TT = max(T) - min(T)
df = 1 / (TT * 5)
norm_sigma = 4 * df
h_norm = (template_nfft * template_nfft.conjugate() / power).sum()
norm_corr = 4 * df / np.sqrt(h_norm.real * norm_sigma)
corr = data_nfft * template_nfft.conjugate() / power
snr = new_ndft(T, corr, frequency) * norm_corr * TT * (len(data_nfft) - 1) / N
snr = np.roll(snr, len(snr) // 2)
plt.figure(7)
plt.plot(np.abs(snr))
plt.show()


