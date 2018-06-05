import numpy as np
import scipy.signal as signal
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class SimulateTimes:
    def __init__(self, n_samples, underlying_delta=None, small_sigma=None,
                 seed=None):
        self.n_samples = n_samples
        self.underlying_delta = self._compute_delta(underlying_delta)
        self.small_sigma = self._compute_epsilon_sigma(small_sigma)
        self.epsilon = self._compute_random(seed)
        self.times = np.zeros(n_samples)

    def _compute_delta(self, underlying_delta):
        if underlying_delta is None:
            underlying_delta = 0.1
        elif isinstance(underlying_delta, complex):
            underlying_delta = underlying_delta.real

        if not isinstance(underlying_delta, (int, float, complex)):
            raise ValueError("'underlying_delta' parameter should be a number")
        return underlying_delta

    def _compute_times(self, time_init):
        if time_init is None:
            time_init = 0
        elif isinstance(time_init, complex):
            logger.warning("using real part of complex number fo the "
                           "initial time")
            time_init = time_init.real
        elif not isinstance(time_init, (int, float)):
            raise ValueError("time_init must be a number, "
                             "got {} instead".format(type(time_init)))
        return time_init + np.arange(self.n_samples) * self.underlying_delta

    def _compute_random(self, seed):
        if isinstance(seed, int):
            np.random.seed(seed)
        return np.random.normal(0, self.small_sigma, self.n_samples)

    def _compute_epsilon_sigma(self, small_sigma):
        if small_sigma is None:
            small_sigma = 0.1 * self.underlying_delta

        if small_sigma > 0.5 * self.underlying_delta:
            logger.warning("using sigma of 50% of underlying delta time")

        elif small_sigma > self.underlying_delta:
            logging.warning("epsilon sigma was bigger that underlying delta,"
                            "changed to 10%")
            small_sigma = 0.1 * self.underlying_delta

        return small_sigma

    def _add_irregularities(self):
        for i in range(self.n_samples):
            self.times[i] += self.epsilon[i]

    def _normalize(self):
        if min(self.times) < 0:
            self.times += abs(min(self.times))

    def _change_spacing(self, range_points: list, gamma):
        self.times[range_points[0]:range_points[1]] *= gamma

    def _outlier(self, break_point, empty_window):
        self.times[break_point:] += empty_window

    def get_times(self, configuration="slight", time_init=None):
        self.times = self._compute_times(time_init)
        if configuration == "slight":
            self._add_irregularities()
            self.times.sort()
            self._normalize()

        elif configuration == "outlier":
            self._outlier(3 * int(self.n_samples / 6),
                          int(self.n_samples / 8) * self.underlying_delta)
            self._add_irregularities()
            self.times.sort()
            self._normalize()

        elif configuration == "change":
            self._change_spacing([2 * int(self.n_samples / 6),
                                  4 * int(self.n_samples / 6)],
                                 3)
            self._add_irregularities()
            self.times.sort()
            self._normalize()

        elif configuration == "mix1":
            self._change_spacing([2 * int(self.n_samples/6),
                                  4 * int(self.n_samples/6)],
                                 3)
            self._outlier(3 * int(self.n_samples/6),
                          int(self.n_samples/8) * self.underlying_delta)
            self._add_irregularities()
            self.times.sort()
            self._normalize()
        else:
            raise ValueError("configuration "
                             "{} not implemented".format(configuration))
        return self.times


class SimulateSignal(SimulateTimes):
    def __init__(self, n_samples, freq, weights=None, underlying_delta=None,
                 small_sigma=None, seed=None, noise_level=None, dwindow=None):
        super().__init__(n_samples, underlying_delta=underlying_delta,
                         small_sigma=small_sigma, seed=seed)
        self.freqs, self.weights = self._compute_freqs(freq, weights)
        self.noise_level = self._compute_noise_level(noise_level)
        self.dwindow_type = self._compute_dwindow_type(dwindow)

    def _compute_dwindow_type(self, dwindow):
        if dwindow is None:
            dwindow = "tukey"

        if dwindow not in ["tukey"]:
            raise ValueError("dwindow '{}' not implemented,".format(dwindow) +
                             " use one of: ['tukey']")

        return dwindow

    def compute_dwindow(self):
        if self.dwindow_type is "tukey":
            return signal.tukey(self.n_samples, alpha=1./8)

    def _compute_noise_level(self, noise_level):
        if noise_level is None:
            noise_level = 0.4
        elif isinstance(noise_level, complex):
            noise_level = noise_level.real

        if not isinstance(noise_level, (int, float, complex)):
            raise ValueError("parameter 'noise_level' should be a number")

        else:
            # we dont want a noise level higher than the sum of weights
            noise_level %= np.array(self.weights).sum()
            print(noise_level)
        return noise_level

    def _compute_freqs(self, freq, weights):
        if freq is None:
            return freq, [1]

        if isinstance(freq, (int, float, complex)):
            freq = [freq.real]
            weights = [1]

        elif isinstance(freq, (np.ndarray, list)):
            if not isinstance(weights, (np.ndarray, list)):
                weights = np.ones(len(freq)) / len(freq)
            else:
                if len(weights) != len(freq):
                    raise ValueError("the number of " +
                                     "frequencies ({})".format(len(freq)) +
                                     "given must match the number of " +
                                     "weights ({}) given".format(len(weights)))

            if np.array(weights).ndim != 1 or np.array(freq).ndim != 1:
                raise ValueError("arrays 'freq' and 'weights' " +
                                 "must be one-dimensional")

        return freq, weights

    def _get_range_of_signal(self, pos_start_peaks, n_peaks):
        pos_start_peaks %= self.n_samples
        time_start = self.times[pos_start_peaks]
        if self.freqs is None:
            self.freqs = [2 / (max(self.times) - time_start)]
            logger.warning("using frequency {} by default".format(self.freqs))
            idx_end = len(self.times) - 1
        else:
            max_period = 1/min(self.freqs)
            time_end = time_start + n_peaks * max_period
            if time_end > max(self.times):
                time_end = max(self.times)
            idx_end = np.abs(self.times - time_end).argmin()

        return pos_start_peaks, idx_end

    def _wave(self, i, times):
        return self.weights[i] * np.sin(2 * np.pi * self.freqs[i] * times)

    def get_noise(self, seed):
        if seed is not None:
            np.random.seed(seed)

        return np.random.normal(0, self.noise_level, self.n_samples)

    def get_data(self, pos_start_peaks=None, n_peaks=None,
                 seed=None, with_noise=False, configuration="slight"):

        _ = self.get_times(configuration=configuration, time_init=0)
        idx_start, idx_end = self._get_range_of_signal(pos_start_peaks,
                                                       n_peaks)
        time_duration = self.times[idx_end] - self.times[idx_start]
        isolated_time = self.times[idx_start:idx_end+1] - self.times[idx_start]
        # create the vector
        data = np.zeros(self.n_samples)
        for i in range(len(self.freqs)):
            data[idx_start:idx_end+1] += self._wave(i, isolated_time)

        if with_noise:
            data += self.get_noise(seed)

        return data * self.compute_dwindow()


