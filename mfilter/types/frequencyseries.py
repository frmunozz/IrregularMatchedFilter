import numpy as np
from mfilter.types.arrays import Array
import matplotlib.pyplot as plt


class FrequencySamples(Array):
    def __init__(self, time: Array, n=None, minimum_frequency=None,
                 maximum_frequency=None, samples_per_peak=1, nyquist_factor=2):

        df = 1 / time.duration / samples_per_peak

        if minimum_frequency is None:
            minimum_frequency = 0.5 * df

        if maximum_frequency is None:
            # bad estimation of nyquist limit
            average_nyq = 0.5 * len(time) / time.duration
            # to fix this bad estimation, amplify the estimated value by 5
            maximum_frequency = nyquist_factor * average_nyq

        frequency = self._frequency(n, df, minimum_frequency,
                                    maximum_frequency)
        super().__init__(frequency, delta=df)

    def _frequency(self, n, df, minimum_frequency, maximum_frequency):
        if n is None:
            n = 1 + int(np.round((maximum_frequency - minimum_frequency) / df))

        return minimum_frequency + df * np.arange(n)

    def check_nsst(self, B):
        return 2 * B < self.end

    def split_by_zero(self):
        idx = self.zero_idx
        if idx is None:
            return None, None
        else:
            return self.data[:idx], self.data[idx+1:]

    @property
    def min_freq(self):
        return self.offset

    @property
    def max_freq(self):
        return self.end
    @property
    def has_zero(self):
        return 0 in self.data
    @property
    def zero_idx(self):
        if self.has_zero:
            return np.where(self.data == 0)[0][0]
        else:
            return None

    @property
    def frequencies(self):
        return self._data

    @property
    def df(self):
        return self._delta


class FrequencySeries(Array):
    def __init__(self, initial_array, frequency=None, times=None,
                 df=None, **kwargs):
        super().__init__(initial_array, delta=df)
        self._freqs = self._validate_freqs(frequency, df, times,
                                           **kwargs)

    @property
    def frequency(self) -> Array:
        return self._freqs

    @property
    def df(self):
        return self.delta

    def _validate_freqs(self, frequency, df, times, **kwargs):
        _boolean_aux = df is not None and times is not None
        if not isinstance(times, Array) and times is not None:
            times = Array(times)
        if frequency is None and _boolean_aux:
            # compute the frequencies
            frequency = FrequencySamples(
                times, n=len(self),
                minimum_frequency=kwargs.get("minimum_frequency"),
                maximum_frequency=kwargs.get("maximum_frequency"),
                samples_per_peak=kwargs.get("samples_per_peak", 1),
                nyquist_factor=kwargs.get("nyquist_factor", 2))

        elif isinstance(frequency, (list, np.ndarray)):
            df = frequency[1] - frequency[0]
            frequency = Array(frequency, delta=df)

        elif not isinstance(frequency, Array):
            raise ValueError("non valid type of 'frequency'")

        return frequency

    def frequency_slice(self, start, end):
        start_idx = np.argmin(np.abs(self._freqs.offset - start))
        end_idx = np.argmin(np.abs(self._freqs.end - end))
        return self.slice_by_indexes(start_idx, end_idx), \
            self._freqs.slice_by_values(start, end)

    def reconstruct(self, reg, times: Array = None):
        if times is not None:
            reg.set_dict(times.data, self._freqs.data)
        return reg.reconstruct(self.data)

    def infft(self):
        pass

    def plot(self):
        plt.figure()
        plt.plot(self._freqs.data, abs(self.data))
        plt.show()
