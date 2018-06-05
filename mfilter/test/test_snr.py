from mfilter.core import SegmentMatchedFilter, get_frequency
from mfilter.implementations.simulate import SimulateSignal
from astropy.stats import LombScargle
import matplotlib.pyplot as plt
import numpy as np

n_samples = 400
freq = [0.0001, 0.005]
weights=[2, 0.1]
config="outlier"
pos_start_peaks = 120
n_peaks = 0.5
simulated = SimulateSignal(n_samples, freq, weights=weights, noise_level=0.2,
                           dwindow="tukey", underlying_delta=60)
times = simulated.get_times(configuration=config)
data = simulated.get_data(pos_start_peaks=pos_start_peaks, n_peaks=n_peaks,
                          with_noise=True,
                          configuration=config)
noise = simulated.get_noise(None)
temp = simulated.get_data(pos_start_peaks=pos_start_peaks, n_peaks=n_peaks,
                          with_noise=False,
                          configuration=config)
freqs = get_frequency(times, nyquist_factor=1)

noise_psd = LombScargle(times, noise).power(freqs, normalization="psd")
plt.figure()
plt.plot(times, temp, 'k')
plt.plot(times, data, '.--')
plt.show()

method = "regression"
regression_method="elasticnet"
prm_regression={"alphas":[0.01], "l1ratio": [0.5]}
mf = SegmentMatchedFilter(times, noise, temp, noise_psd=noise_psd)
snr = mf.compute_snr(method=method, normalize="off",
                     regression_method=regression_method,
                     frequencies=freqs, prm_regresion=prm_regression)
plt.figure()
plt.plot(times, np.abs(snr))
plt.show()
snr = mf.compute_snr(method=method, normalize="noise",
                     regression_method=regression_method,
                     frequencies=freqs, prm_regresion=prm_regression)
plt.figure()
plt.plot(times, np.abs(snr))
plt.show()
snr = mf.compute_snr(method=method, normalize="template",
                     regression_method=regression_method,
                     frequencies=freqs, prm_regresion=prm_regression)
plt.figure()
plt.plot(times, np.abs(snr))
plt.show()
