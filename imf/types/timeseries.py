import numpy as np
import matplotlib.pyplot as plt
from imf.types.arrays import Array
from imf.types.frequencyseries import FrequencySamples, FrequencySeries
from imf.transform.transform import FourierTransformer


class TimeSamples(Array):
    def __init__(self, initial_array=None, n=None, delta=None,
                 struct="slight", clear=True, **kwargs):
        """
        Time Samples Class,
            Allow creation fo time samples array which contains the
            information about the samples used on a time series.

            It can receive an array-like with the information about the
            time samples or the parameters necessary to create a time samples.

            There are two main options to create a time samples, one is to create
            a evenly sampled and the other is to create an unevenly sampled times,
            and there are several options for unevenly sampled times, like 'slightly',
            'outlier', 'change' or 'automix'.

        :param initial_array:       an array of the existing time samples. In the case
                                    of being 'None', it will create a new time samples.
        :param n:                   number of samples, parameter used to create a new
                                    time samples.
        :param delta:               base time step, parameter used to create a new time
                                    sample.
        :param struct:              struct type, parameter used to create a new time
                                    sample, options are:
                                    -slightly: for small irregularities in time sample.
                                    - outlier: is 'slightly' plus a big gap in the time samples.
                                    - change: is 'slightly' plus a change in time step.
                                    - automix: is 'slightly' + 'outlier' + 'change' on
                                                random positions.
        :param clear: True to create a new time samples. False to modify the previously one.
        :param kwargs: additional paramters for the creation of time samples
        """
        if initial_array is None:
            if delta <= 0:
                raise ValueError("need to receive a valid delta of times")

            if "regular" in struct:
                offset = kwargs.get("offset", 0)
                initial_array = offset + np.arange(n) * delta
            else:
                arr = IrregularTimeSamples(n, delta)
                initial_array = arr.compute(struct=struct, clear=clear, **kwargs)

        else:
            if isinstance(initial_array, TimeSamples):
                initial_array = initial_array.data

        super().__init__(initial_array)

    @property
    def average_fs(self):
        """
        :return: averaga sampling rate
        """
        return len(self._data) / self.duration

    @property
    def duration(self):
        """
        :return: duration of the observed time interval
        """
        return self._data.max() - self._data.min()

    def shifted(self, shift_by):
        """
        shift the time interval by specific value.

        :param shift_by: value to shift
        :return: the sifted time.
        """
        return self - shift_by

    def _return(self, ary, **kwargs):
        """
        modification of the base _return method to give
        a TimeSamples object.

        :param ary:
        :param kwargs:
        :return:
        """
        return TimeSamples(ary)


class IrregularTimeSamples(object):
    def __init__(self, n: int, dt: float, grid: np.ndarray=None):
        """
        Creation of Irregular time samples (for unevenly time series),
            starting from a regular array of time samples, it will
            create an irregular one adding random variations to the array.

            there are several options of the type of irregularities to introduce.

        :param n:           the number of samples to create.
        :param dt:          the initial time step to use.
        :param grid:        a grid to use as the initial array, if is None,
                            it will create a regular grid.

        """
        if dt <= 0:
            raise ValueError("'dt' need to be positive and non zero")

        if grid is None:
            grid = np.arange(n) * dt

        if n != len(grid):
            raise ValueError("the 'grid' used need to be of length 'n'")

        self.n = n
        self.dt = dt
        self.t = grid

    def compute(self, struct="slight", clear=True, **kwargs):
        """

        compute the time samples given the parameters.

        :param struct:      the type of struct to use, the options are:
                                - slightly: for small irregularities.
                                - outlier: for a big gap.
                                - change: for a change in time step.
                                - automix: for a combination of all types.
        :param clear: True if we start from a new regular array.
        :param kwargs: additional parameters.
        :return: the time samples (array-like)
        """
        kwargs = self._set_kwargs(kwargs)
        if clear:
            self.t = np.arange(self.n) * self.dt

        if "slight" in struct:
            self._base(**kwargs)

        elif "outlier" in struct:
            self._outlier(**kwargs)

        elif "change" in struct and "spacing" in struct:
            self._change_spacing(**kwargs)

        elif "automix" in struct:
            self._auto_mix()

        return self.t

    def _set_kwargs(self, kwargs):
        """
        set all the parameters to be used,
        they come in kwargs form to simplify functions head.

        If a parameter doesnt come defined in kwargs it will
        be defined with a default value.

        :param kwargs:
        :return:
        """
        _ = kwargs.setdefault("offset", 0)
        _ = kwargs.setdefault("sigma", 0.05)
        epsilon = np.random.normal(0, kwargs.get("sigma") * self.dt, self.n)
        _ = kwargs.setdefault("epsilon", epsilon)
        break_point = np.random.randint(0, self.n)
        _ = kwargs.setdefault("break_point", break_point)
        empty_window = np.random.random() * self.n * self.dt
        _ = kwargs.setdefault("empty_window", empty_window)
        start_point = np.random.randint(0, self.n)
        _ = kwargs.setdefault("start_point", start_point)
        _ = kwargs.setdefault("end_point", self.n)
        return kwargs

    def _add_irregularities(self, **kwargs):
        """
        add irregularities to the time samples.

        :param kwargs:
        """
        for i in range(self.n):
            self.t[i] += kwargs.get("epsilon")[i]

    def _normalize(self, offset=0):
        """
        normalize the time samples to start from zero and then
        add an offset

        :param offset: value to add
        """
        if self.t.min() < offset:
            self.t += offset - abs(self.t.min())

    def _base(self, **kwargs):
        """
        create the basic irregular time samples which is
        an array with irregularities and normalized to 0
        plus offset.

        :param kwargs:
        """
        self._add_irregularities(**kwargs)
        self._normalize(offset=kwargs.get("offset"))

    def _outlier(self, **kwargs):
        """
        add to the small irregularities a big gap
        starting from the break_point. The size of
        the gap will be given by 'empty_window'

        :param kwargs:
        """
        self.t[kwargs.get("break_point"):] += kwargs.get("empty_window")
        self._base(**kwargs)

    def _change_spacing(self, **kwargs):
        """
        modify the base regular array by changing the
        time step starting from 'start_point' to 'end_point'
        and then create the base irregular time samples.

        :param kwargs:
        """
        start_point = kwargs.get("start_point")
        end_point = kwargs.get("end_point")
        self.t[start_point:end_point] *= kwargs.get("gamma")
        self._base(**kwargs)

    def _auto_mix(self):
        """
        create a complete irregular time samples with
        small irregularities, a big gap and change in
        time steps.

        """
        gamma = np.random.choice([0.5, 1, 1.5, 2])
        offset = np.random.uniform(0, self.n * self.dt)
        empty_window = np.random.uniform(0, self.n * self.dt * 0.5)
        self.t.sort()
        self.t[2 * (self.n // 5):4 * (self.n // 5)] *= gamma
        self.t[3 * (self.n // 5):] += empty_window
        self._add_irregularities(epsilon=np.random.normal(0,
                                                          0.05 * self.dt,
                                                          self.n))
        self.t.sort()
        self._normalize(offset=offset)


class TimeSeries(Array):
    def __init__(self, initial_array, times=None, delta_t=None, regular=False,
                 dtype=None, **kwargs):
        """
        Time Series object,
            create a TimeSeries object which contains information
            about the data and their related time samples.

            It need as input an array-like with the timeSeries data,
            and the time samples, which can be directly a TimeSamples
            object or the parameters necessary to create a TimeSamples.

        :param initial_array:       array-like with the data of the TimeSeries.
        :param times:               TimeSamples object with the time of the TimeSeries.
        :param delta_t:             time step for the TimeSamples.
        :param regular:             type of the TimeSamples.
        :param dtype:               data type of the array-like data.
        :param kwargs:              extra parameters for the creation of TimeSamples.
        """
        if len(initial_array) < 1:
            raise ValueError('initial_array must contain at least one sample.')

        if times is None:
            if delta_t is None:
                try:
                    delta_t = initial_array.delta_t
                except AttributeError:
                    raise TypeError('must provide either an initial_array with'
                                    ' a delta_t attribute, or a value for '
                                    'delta_t')
            times = TimeSamples(n=len(initial_array), delta=delta_t,
                                regular=regular, **kwargs)

        if not isinstance(times, TimeSamples):
            raise ValueError("time must be a TimesSamples object")

        super().__init__(initial_array, dtype=dtype)
        self._times = times

    def __eq__(self, other):
        """
        two time series are going to be equal when they have exactly same
        sampling times, the values of every time doesnt need to be the same
        :param other: the other time series
        """
        return self.times == other.times

    @property
    def times(self):
        return self._times

    @property
    def duration(self):
        return self._times.duration

    @property
    def start_time(self):
        return self._times.min()

    @property
    def end_time(self):
        return self._times.max()

    @property
    def epoch(self):
        return self._times.min()

    @property
    def average_fs(self):
        return self._times.average_fs

    def _getslice(self, index):
        return self._return(self._data[index],
                            times=TimeSamples(initial_array=self._times[index]))

    def get_time_slice(self, start_time, end_time):
        start_idx = np.abs(self._times - start_time).argmin()
        end_idx = np.abs(self._times - end_time).argmin()
        return self._getslice(slice(start_idx, end_idx))

    def _return(self, ary, **kwargs):
        times = kwargs.get("times", self._times)
        return TimeSeries(ary, times=times)

    def delete(self, idxs:slice):
        self._data = np.delete(self._data,idxs)
        self._times = TimeSamples(initial_array=np.delete(self._times, idxs))

    def psd(self, frequency_grid: FrequencySamples, norm='standard'):
        """
        calculate the power spectral density of this time series.

        it uses the Lomb-Scargle periodogram

        :param frequency_grid: FrequencySamples object
        :return: FrequencySeries
        """
        return frequency_grid.lomb_scargle(self.times, self.data, norm=norm)

    def to_frequencyseries(self, transformer: FourierTransformer):
        """
        calculate the Fourier Transform of the TimeSeries and create a
        FrequencySeries.

        The transform is done by a Transformer which can use Fast Fourier Transform
        or Regressions for the Fourier Transform.

        :param transformer:
        :return:
        """
        if self.kind == 'complex':
            raise TypeError("transform only work with real timeSeries data")

        tmp = transformer.backward(self)
        freqs = transformer.get_frequency(N=len(self))
        return FrequencySeries(tmp, frequency_grid=freqs, epoch=self.epoch)

    # TODO: NOT IMPLEMENTED
    # def match(self, other, transformer: FourierTransformer, psd=None):
    #     """
    #     perform Matched Filter against another TimeSeries and return their Match time.
    #
    #     :param other:           other TimeSeries.
    #     :param transformer:     Transformer used for Fourier Transform
    #     :param psd:             Power Spectral Density of the noise. If is none,
    #                             Compute only Matched Filter, if is not None, perform
    #                             Matched Filter with Whitening Filter.
    #     :return: TimeSeries object with the SNR data on the TimeSamples used in the original data.
    #     """
    #     return self.to_frequencyseries(transformer).match(other, psd=psd)
