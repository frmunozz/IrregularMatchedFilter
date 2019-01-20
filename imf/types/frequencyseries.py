"""
Provides a class representing a frequency series.
"""

import numpy as np
from imf.types.arrays import Array
from astropy.stats import LombScargle
import matplotlib.pyplot as plt
import scipy.signal as signal


class FrequencySamples(Array):
    def __init__(self, initial_array=None, input_time=None,
                 minimum_frequency=None, maximum_frequency=None,
                 samples_per_peak=None, nyquist_factor=2, n_samples=None,
                 df=None):
        """
        class for creation of frequency samples on regular grid, on custom
        values, using an oversampling factor.

        :param input_time:          Array or array-like
                                    times of the original data
        :param minimum_frequency:   scalar (real)
                                    minimum frequency to compute, default 0
        :param maximum_frequency:   scalar (real)
                                    maximum frequency to compute
        :param samples_per_peak:    integer
                                    number of oversampling per peak, default 5
        :param nyquist_factor:      scalar (real)
                                    value used for estimation of maximum freq.
        :param n_samples:           integer
                                    number of frequencies to compute.
        """
        if initial_array is None:
            if not isinstance(input_time, (np.ndarray, list, Array)):
                raise ValueError("input_time must be an Array or array-like")

            if samples_per_peak is None:
                samples_per_peak = 5

            duration = input_time.max() - input_time.min()

            df = 1 / duration / samples_per_peak

            if minimum_frequency is None:
                minimum_frequency = 0

            if maximum_frequency is None:
                if n_samples is None:
                    # bad estimation of nyquist limit
                    average_nyq = 0.5 * len(input_time) / duration
                    # to fix this bad estimation, amplify by nyquist_factor
                    maximum_frequency = nyquist_factor * average_nyq
                    freq_duration = maximum_frequency - minimum_frequency
                    n_samples = 1 + int(round(freq_duration / df))
            else:
                freq_duration = maximum_frequency - minimum_frequency
                n_samples = 1 + int(round(freq_duration / df))

            initial_array = minimum_frequency + df * np.arange(n_samples)
        else:
            if isinstance(initial_array, FrequencySamples):
                df = initial_array.basic_df
                samples_per_peak = initial_array.samples_per_peak
                initial_array = initial_array.data
            else:
                if df is None:
                    df = initial_array[1] - initial_array[0]
                if samples_per_peak is None:
                    samples_per_peak = 1

        self._df = df
        self._n_per_peak = samples_per_peak
        super().__init__(initial_array)

    def check_nsst(self, B):
        """
        check if the Nyquist-Shannon sampling theorem is satisfied

        :param B:       Maximum frequency of interest
        :return:        True if the NSST is satisfied
        """
        return 2 * B < self.max()

    def split_by_zero(self):
        """
        split the frequency grid in order to avoid 0 frequency for computation
        of Lomb-Scargle periodogram

        :return:
        """
        idx = self.zero_idx
        if idx is None:
            return None, None
        else:
            return self._data[:idx], self._data[idx+1:]

    def _return(self, ary, **kwargs):
        if isinstance(ary, FrequencySamples):
            return ary

        return FrequencySamples(ary, df=self.basic_df,
                                samples_per_peak=self.samples_per_peak)

    @property
    def has_zero(self):
        return 0 in self._data

    @property
    def zero_idx(self):
        if self.has_zero:
            return np.where(self._data == 0)[0][0]
        else:
            return None

    @property
    def df(self):
        return self._df

    @property
    def samples_per_peak(self):
        return self._n_per_peak

    @property
    def basic_df(self):
        return self._df * self._n_per_peak

    def lomb_scargle(self, times, data, package="astropy", norm="standard", weighted=False, windowed=False):
        """
        compute the Lomb-Scargle periodogram using astropy package.

        :param times:
        :param data:
        :param package:
        :param norm:
        """
        if package is "astropy":
            lomb = LombScargle(times, data)
        else:
            raise ValueError("for now we only use astropy package to compute"
                             "lomb-scargle periodogram")
        window = 1
        W = 1
        if windowed:
            window = signal.windows.tukey(len(data), alpha=1. / 8)
        data *= window
        if windowed and weighted:
            W = (window ** 2).sum() / len(window)

        if self.has_zero:
            zero_idx = self.zero_idx
            psd = np.zeros(len(self._data))
            if zero_idx == 0:
                psd[1:] = lomb.power(self._data[1:])
                psd[0] = min(psd[1:])
            else:
                neg_freq, pos_freq = self.split_by_zero()
                right_psd = lomb.power(pos_freq)
                left_psd = lomb.power(np.abs(neg_freq[::-1]))
                psd[:zero_idx] = left_psd[::-1]
                psd[zero_idx] = min(psd[:zero_idx])
                psd[zero_idx+1:] = right_psd
        else:
            psd = lomb.power(np.abs(self._data), normalization=norm)

        # psd[psd < 0] = 0.000001

        return FrequencySeries(psd / W, frequency_grid=self, epoch=times.min())

    def lomb_welch(self, times, data, data_per_segment, over,
                   norm='standard', weighted=True, windowed=True):
        """
        compute the Lomb-Scargle average periodogram usign welch computation,
        this method is not developed yet, idea comes from paper:

        Tran Thong, James McNames and Mateo Aboy, "Lomb-Welch Periodogram for
        Non-uniform Sampling", IEEE-EMBS, 26th annual International conference,
        Sep 2004.

        :param over:
        :param data_per_segment:
        :param norm:
        :param weighted:
        :param windowed:
        :param times:
        :param data:
        """
        # import pdb
        # pdb.set_trace()
        psd = FrequencySeries(np.zeros(len(self)), frequency_grid=self,
                              epoch=data.epoch)

        counter = 0
        n = 0
        W = 1
        window = 1
        while n < len(data) - data_per_segment:
            aux_timeseries = data.get_time_slice(times[n],
                                                 times[n + data_per_segment])
            if windowed:
                window = signal.windows.tukey(len(aux_timeseries), alpha=1. / 8)
            aux_timeseries *= window
            if weighted and windowed:
                W = (window ** 2).sum() / len(window)
            psd += (aux_timeseries.psd(self, norm=norm) / W)
            n += int(data_per_segment * over)
            counter += 1

        aux_timeseries = data.get_time_slice(
            times[len(times) - data_per_segment - 1], times[len(times) - 1])
        if windowed:
            window = signal.windows.tukey(len(aux_timeseries), alpha=1. / 8)
        aux_timeseries *= window
        if weighted and windowed:
            W = (window ** 2).sum() / len(window)
        psd += (aux_timeseries.psd(self, norm=norm) / W)
        counter += 1
        psd /= counter
        return psd


class FrequencySeries(Array):
    def __init__(self, initial_array, frequency_grid=None, delta_f=None,
                 minimum_frequency=None, samples_per_peak=None, epoch=None,
                 dtype=None):
        if len(initial_array) < 1:
            raise ValueError('initial_array must contain at least one sample.')

        if frequency_grid is None:
            if delta_f is None:
                try:
                    delta_f = initial_array.basic_df
                except AttributeError:
                    raise TypeError("must provide either an initial_array "
                                    "with a delta_f attribute, or a value "
                                    "of delta_f")
            if delta_f <= 0:
                raise ValueError('delta_f must be a positive number')

            if minimum_frequency is None:
                try:
                    minimum_frequency = initial_array.min_freq
                except AttributeError:
                    minimum_frequency = 0

            if samples_per_peak is None:
                try:
                    samples_per_peak = initial_array.samples_per_peak
                except AttributeError:
                    samples_per_peak = 1

            # generate synthetic input time of same size as initial_array
            input_times = np.linspace(0, 1/delta_f, len(initial_array))
            frequency_grid = FrequencySamples(input_times,
                                              minimum_frequency=minimum_frequency,
                                              samples_per_peak=samples_per_peak,
                                              n_samples=len(initial_array))
        else:
            try:
                _ = frequency_grid.basic_df
                _ = frequency_grid.min()
                _ = frequency_grid.samples_per_peak
            except AttributeError:
                raise TypeError("must provide either a FrequencySamples object"
                                "as frequency_grid or the parameters necessary"
                                "to compute a frequency grid")

        if epoch is None:
            try:
                epoch = initial_array.epoch
            except AttributeError:
                raise TypeError("must provide either an initial_array with an"
                                "epoch attribute, or a value of epoch")

        assert len(initial_array) == len(frequency_grid)

        super().__init__(initial_array, dtype=dtype)
        self._freqs = frequency_grid
        self._epoch = epoch

    @property
    def frequencies(self):
        return self._freqs

    @property
    def df(self):
        return self._freqs.df

    @property
    def basic_df(self):
        return self._freqs.basic_df

    @property
    def max_freq(self):
        return self._freqs.max()

    @property
    def min_freq(self):
        return self._freqs.min()

    @property
    def samples_per_peak(self):
        return self._freqs.samples_per_peak

    @property
    def epoch(self):
        return self._epoch

    @property
    def start_time(self):
        return self._epoch

    @property
    def end_time(self):
        return self._epoch + self.duration

    @property
    def duration(self):
        return 1 / self.basic_df

    def __eq__(self, other):
        """
        two FrequencySeries are equals if their values are the same,
        their frequency steps are the same and the epoch is the same.

        :param other:
        :return:
        """
        if super(FrequencySeries, self).__eq__(other):
            return (self._epoch == other.epoch
                    and self.basic_df == other.basic_df)
        else:
            return False

    def to_timeseries(self, transformer, **kwargs):
        """
        calculate the Fourier Transform of the FrequencySeries and create a
        TimeSeries.

        The transform is done by a Transformer which can use Fast Fourier Transform
        or Regressions for the Fourier Transform.

        :param transformer:         Transformer Object
        :param kwargs:              additional parameters for the Fourier Transform.
        :return:                    TimeSeries object of the Fourier Transform
        """
        from imf.types.timeseries import TimeSeries

        tmp = transformer.forward(self, kwargs)
        return TimeSeries(tmp, times=transformer.get_times())

    # TODO: NOT IMPLEMENTED
    # def match(self, other, psd=None, tol=0.1):
    #     from imf.types import TimeSeries
    #     from imf.filter import match
    #
    #     if isinstance(other, TimeSeries):
    #         if abs(other.duration / self.duration - 1) > tol:
    #             raise ValueError("duration of times is higher than given "
    #                              "tolerance")
    #         other = other.to_frequencyseries()
    #
    #     assert len(other) == len(self)
    #
    #     if psd is not None:
    #         assert len(psd) == len(self)
    #
    #     return match(self, other, psd=psd)
    #
    # def split_values(self):
    #     return np.hstack((self.real, self.imag))
