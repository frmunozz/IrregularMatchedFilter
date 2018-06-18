
from mfilter.implementations.simulate import SimulateSignal
from mfilter.implementations.regressions import RidgeRegression, ElasticNetRegression, Dictionary
from mfilter.types.timeseries import TimeSeries
from mfilter.types.frequencyseries import FrequencySamples
from mfilter.core import MatchedFilterRegression
import matplotlib.pyplot as plt
from astropy.stats import LombScargle
import numpy as np


n_samples = 50
freq = [0.00007, 0.001, 0.01]
weights=[1, 0.4, 0.2]
config="mix1"
pos_start_peaks = 0
n_peaks = 1
simulated = SimulateSignal(n_samples, freq, weights=weights, noise_level=0.2,
                           dwindow="tukey", underlying_delta=50)
times = simulated.get_times(configuration=config)
data = simulated.get_data(pos_start_peaks=pos_start_peaks, n_peaks=n_peaks,
                          with_noise=True,
                          configuration=config)
noise = simulated.get_noise(None)
temp = simulated.get_data(pos_start_peaks=0, n_peaks=1, with_noise=False,
                          configuration=config)

temp2 = simulated.get_data(pos_start_peaks=n_samples//2, n_peaks=0.5,
                           configuration=config)
plt.figure()
plt.plot(times, data, label='data')
plt.plot(times, temp, label='template 1')
plt.plot(times, temp2, label='template 2')
plt.plot(times, noise, label="noise")
plt.legend()
plt.show()

data = TimeSeries(data, times=times, regular=False)
temp = TimeSeries(temp, times=times, regular=False)
temp2 = TimeSeries(temp2, times=times, regular=False)
noise = TimeSeries(noise, times=times, regular=False)
samples_per_peak = 5
df = 1 / data.times.duration / samples_per_peak
max_freq = 2 * max(freq)

freq = FrequencySamples(data.times,
                        minimum_frequency=0,
                        maximum_frequency=max_freq)
lomb = LombScargle(times, noise.data)

if freq.has_zero:
    zero_idx = freq.zero_idx
    print(min(freq.data), max(freq.data), zero_idx, len(freq))
    neg_freq, pos_freq = freq.split_by_zero()
    right_psd = lomb.power(pos_freq)
    left_psd = lomb.power(np.abs(neg_freq))

    psd = np.zeros(len(freq))
    psd[:zero_idx] = left_psd
    psd[zero_idx] = 0.000001
    psd[zero_idx+1:] = right_psd
else:
    psd = lomb.power(np.abs(freq.data))

sigma_square = np.sum(psd**2)
print("variance is: ", sigma_square)
norm = (2 * np.pi * sigma_square)**(n_samples/2)
#norm = 1

# probabilities of the original datas
prob_data = np.exp(- np.sum(data.data) / (2 * sigma_square)) / norm
prob_noise = np.exp(-np.sum(noise.data) / (2 * sigma_square)) / norm
print("probs of the data: ", prob_data, prob_noise)

# plt.figure()
# plt.plot(freq.data, psd)
# plt.show()

reg = RidgeRegression(alpha=0.01, solver='auto')
# reg = ElasticNetRegression(alpha=0.01, l1_ratio=0.5)

mfr = MatchedFilterRegression(data,temp, reg)
snr = mfr.snr(psd=psd, frequency=freq)
snr.roll(len(snr)//2)
E = np.sum(snr.data**2)
L = len(snr.times)
prob = np.exp(-E / (2 * sigma_square)) / norm

mfr2 = MatchedFilterRegression(noise, temp, reg)
snr2 = mfr2.snr(psd=psd, frequency=freq)
snr2.roll(len(snr2)//2)
E2 = np.sum(snr2.data**2)
prob2 = np.exp(-E2 / (2 * sigma_square)) / norm

mfr3 = MatchedFilterRegression(data, temp2, reg)
snr3 = mfr3.snr(psd=psd, frequency=freq)
snr3.roll(len(snr3)//2)
E3 = np.sum(snr3.data**2)
prob3 = np.exp(-E3 / (2 * sigma_square)) / norm

mfr4 = MatchedFilterRegression(noise, temp2, reg)
snr4 = mfr4.snr(psd=psd, frequency=freq)
snr4.roll(len(snr4)//2)
E4 = np.sum(snr4.data**2)
prob4 = np.exp(-E4 / (2 * sigma_square)) / norm

print("probs: ", prob, prob2, prob3, prob4)

plt.figure()
plt.plot(snr.times.data - snr.times.data[len(snr)//2], snr.data,
         label="SNR of data with temp1, prob: {}".format(round(prob, 4)))
plt.plot(snr3.times.data - snr3.times.data[len(snr2)//2], snr3.data,
         label="SNR of data with temp2, prob: {}".format(round(prob3, 4)))
plt.legend()
plt.show()

plt.figure()
plt.plot(snr2.times.data - snr2.times.data[len(snr2)//2], snr2.data,
         label="SNR of noise with temp1, prob: {}".format(round(prob2, 4)))
plt.plot(snr4.times.data - snr4.times.data[len(snr4)//2], snr4.data,
         label="SNR of noise with temp1, prob: {}".format(round(prob4, 4)))

plt.legend()
plt.show()


