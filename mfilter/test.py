from mfilter import *
import numpy as np
import matplotlib.pyplot as plt




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
s = np.zeros(N)
template = Templates()
s[int(N/4): int(N/2)] = template.compute_template(T[int(N/4): int(N/2)])
# s = np.sin(freq_of_sin * 2 * np.pi * T)
#  apply noise
np.random.seed(1)
noise = np.random.normal(0, 0.3, N)
data = s + noise
plt.figure(0)
plt.plot(T, data, alpha=0.5)
plt.plot(T, s, "k.-")
plt.show()

t_seg = (max(T) - min(T))
mf = core.MatchedFilter(T, data, template=template,
                        prm_segment={'ovlp_fact': 0.5, 't_segment': t_seg})
s_t, s_data, s_temp = mf.compute_nfft()
plt.figure(2)
plt.plot(s_t, s_data)
plt.plot(s_t, s_temp)
plt.show()

s_t, s_data, s_temp = mf.compute_nfft()
plt.figure(3)
plt.plot(s_t, s_data)
plt.plot(s_t, s_temp)
plt.show()

s_t, s_data, s_temp = mf.compute_nfft()
plt.figure(4)
plt.plot(s_t, s_data)
plt.plot(s_t, s_temp)
plt.show()


s_t, s_data, s_temp = mf.compute_nfft()
plt.figure(4)
plt.plot(s_t, s_data)
plt.plot(s_t, s_temp)
plt.show()