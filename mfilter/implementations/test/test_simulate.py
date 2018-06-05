from mfilter.implementations.simulate import SimulateSignal
import matplotlib.pyplot as plt

# testing that implementation doesn't fail
n_samples = 100
simulated = SimulateSignal(n_samples, 2, noise_level=0.25, underlying_delta=0.01)

config = "outlier"
times = simulated.get_times(configuration=config)
data = simulated.get_data(pos_start_peaks=20, n_peaks=1, with_noise=True,
                          configuration=config)
plt.figure()
plt.plot(times, data, '.--')
plt.show()
