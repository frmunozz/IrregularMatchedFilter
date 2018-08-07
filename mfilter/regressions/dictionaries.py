
from mfilter.types import TimesSamples, FrequencySamples
import numpy as np


def _split_fourier_dict(F):
    return np.hstack((F.real, F.imag))
    # return np.vstack((F.real, F.imag))


class Dictionary(object):
    def __init__(self, times, frequencies):
        if not isinstance(times, TimesSamples):
            times = TimesSamples(initial_array=times)

        if not isinstance(frequencies, FrequencySamples):
            frequencies = FrequencySamples(initial_array=frequencies)

        self._t = times
        self._f = frequencies
        self._dict = self.compute_dict(times=times, frequency=frequencies)

    def compute_dict(self, times=None, frequency=None):
        if isinstance(times, TimesSamples):
            self._t = times

        if isinstance(frequency, FrequencySamples):
            self._f = frequency

        matrix = self._t.value.reshape(-1, 1) * self._f.value
        return np.exp(2j * np.pi * matrix)
        #matrix = self._t.value * self._f.value.reshape(-1, 1)
        #return np.exp(-2j * np.pi * matrix) / len(self._t)

    @property
    def frequency(self):
        return self._f

    @property
    def time(self):
        return self._t

    @property
    def matrix(self):
        return self._dict

    @property
    def splited_matrix(self):
        return _split_fourier_dict(self._dict)

    @property
    def df(self):
        return self._f.df

    def shape(self, splited=True):
        if splited:
            return self.splited_matrix.shape
        else:
            return self._dict.shape
