from gatspy.periodic import LombScargleFast
from scipy import signal
from astropy.stats import LombScargle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-paper')
import time
"""
comparison between many implementations of lomb-scargle periodogram
"""
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

t_i = time.time()
frequency, power = LombScargle(T, data).autopower()
t_f1 = time.time()
model = LombScargleFast().fit(T, data, None)
periods, power2 = model.periodogram_auto(nyquist_factor=max(frequency))
t_f2 = time.time()
pgram = signal.lombscargle(T, data, frequency, normalize=True)
t_f3 = time.time()

plt.figure(1)
plt.plot(frequency, power, 'r--', label="LS from astropy, time: {}".format(round(t_f1-t_i, 3)))
plt.plot(1 / periods, power2, 'g', alpha=0.6, label="LS from gatspy, time: {}".format(round(t_f2-t_f1, 3)))
plt.plot(frequency, pgram, 'b', label="LS from scipy, time: {}".format(round(t_f3-t_f2, 3)))
plt.xlim([0, 200])
plt.title("Lomb-Scargle periodogram comparison for {} points".format(N))
plt.xlabel("frequency [Hz]")
plt.ylabel("Lomb-Scargle Power")
plt.axvline(freq_of_sin, color='k', linestyle='solid', label="real frequency expected")
plt.axvline(freq_of_sin * 2 * np.pi, color='k', alpha=0.5, linestyle='solid', label="real angular frequency expected")
plt.legend()
plt.show()

"""
at first sight the implementation from astropy seems to be the most faster but its necessary to run 
several repetitions for different numbers of points to see exactply which is more faster, for know 
this is not necessary to do
"""
