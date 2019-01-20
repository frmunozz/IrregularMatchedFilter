
from imf.types import TimeSamples, FrequencySamples
import numpy as np


def _split_fourier_dict(F):
    """
    take a Dictionary of the Fourier Transform (the Fourier Matrix) and
    split it in real and imaginary part, generating a new matrix
    where first we have all the real parts and then we have all the
    imaginary parts.

    :param F:           Dictionary Object of the Fourier Matrix.
    :return:            Array-like matrix with the Fourier Matrix
                        split by real and imaginary part.
    """
    return np.hstack((F.real, F.imag))


class Dictionary(object):
    def __init__(self, times, frequencies):
        """
        Dictionary class,
            For the creation of Dictionaries used in the Regression method,
            in our case, we need a Dictionary of cosines and sines usually called
            Fourier Matrix. For that, we need the times used in the TimeSeries and
            the frequencies where we want to compute the Fourier coefficients
            (Fourier Transform).

        :param times: TimeSamples from the TimeSeries
        :param frequencies: FrequencySamples for the Fourier Transform.
        """
        if not isinstance(times, TimeSamples):
            times = TimeSamples(initial_array=times)

        if not isinstance(frequencies, FrequencySamples):
            frequencies = FrequencySamples(initial_array=frequencies)

        self._t = times
        self._f = frequencies
        self._dict = self.compute_dict(times=times, frequency=frequencies)

    def compute_dict(self, times=None, frequency=None):
        """
        compute the matrix dictionary using the times and frequency given.

        This function also accept new times and/or frequencies, but by default
        it will use the times and frequencies given in construction.

        :param times: a new TimeSamples to use
        :param frequency: a new FrequencySamples to use
        :return: Array-like Matrix with the dictionary values.
        """
        if isinstance(times, TimeSamples):
            self._t = times

        if isinstance(frequency, FrequencySamples):
            self._f = frequency

        matrix = self._t.data.reshape(-1, 1) * self._f.data
        return np.exp(2j * np.pi * matrix)

    @property
    def frequency(self):
        """
        :return: frequencies used to create the dictionary.
        """
        return self._f

    @property
    def time(self):
        """
        :return: times used to create the dictionary.
        """
        return self._t

    @property
    def matrix(self):
        """
        :return: the original dictionary (complex elements).
        """
        return self._dict

    @property
    def splited_matrix(self):
        """
        :return: a split of the dictionary (real elements).
        """
        return _split_fourier_dict(self._dict)

    @property
    def df(self):
        """
        :return: frequency step of the frequencies uses to create the dictionary
        """
        return self._f.df

    def shape(self, split_matrix=True):
        """
        give the shape of the dictionary (matrix), depending
        on 'split_matrix' it can give the shape of the Fourier
        Matrix (with complex elements) or the split of the Fourier
        Matrix (cosine and sine part).

        :param split_matrix:    True if you want the shape of the split if
                                the Fourier Matrix, false if you just want the
                                shape of the Fourier Matrix.
        :return:                A tuple with the shape.
        """
        if split_matrix:
            return self.splited_matrix.shape
        else:
            return self._dict.shape
