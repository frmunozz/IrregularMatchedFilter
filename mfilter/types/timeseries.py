import numpy as np
from mfilter.types.arrays import Array
from mfilter.types.frequencyseries import FrequencySamples, FrequencySeries
import matplotlib.pyplot as plt
from mfilter.transform.transform import FourierTransform
# from pynfft import NFFT, Solver


class TimesSamples(Array):
    def __init__(self, initial_array=None, n=None, delta=None,
                 struct="slight", clear=True, **kwargs):
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
            if isinstance(initial_array, TimesSamples):
                initial_array = initial_array.value

        super().__init__(initial_array)


    @property
    def average_fs(self):
        return len(self._data) / self.duration

    @property
    def duration(self):
        return self._data.max() - self._data.min()

    def shifted(self, shift_by):
        return self - shift_by


class IrregularTimeSamples(object):
    def __init__(self, n: int, dt: float, grid: np.ndarray=None):
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
        for i in range(self.n):
            self.t[i] += kwargs.get("epsilon")[i]

    def _normalize(self, offset=0):
        if self.t.min() < offset:
            self.t += offset - abs(self.t.min())

    def _base(self, **kwargs):
        self._add_irregularities(**kwargs)
        self._normalize(offset=kwargs.get("offset"))

    def _outlier(self, **kwargs):
        self.t[kwargs.get("break_point"):] += kwargs.get("empty_window")
        self._base(**kwargs)

    def _change_spacing(self, **kwargs):
        start_point = kwargs.get("start_point")
        end_point = kwargs.get("end_point")
        self.t[start_point:end_point] *= kwargs.get("gamma")
        self._base(**kwargs)

    def _auto_mix(self):
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
            times = TimesSamples(n=len(initial_array), delta=delta_t,
                                 regular=regular, **kwargs)

        if not isinstance(times, TimesSamples):
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
                            times=TimesSamples(initial_array=self._times[index]))

    def get_time_slice(self, start_time, end_time):
        start_idx = np.abs(self._times - start_time).argmin()
        end_idx = np.abs(self._times - end_time).argmin()
        return self._getslice(slice(start_idx, end_idx))

    def _return(self, ary, **kwargs):
        times = kwargs.get("times", self._times)
        return TimeSeries(ary, times=times)

    def delete(self, idxs:slice):
        self._data = np.delete(self._data,idxs)
        self._times = TimesSamples(initial_array=np.delete(self._times, idxs))

    def psd(self, frequency_grid: FrequencySamples, norm='standard'):
        """
        calculate the power spectral density of this time series.

        For now, it uses the Lomb-Scargle periodogram but in future should
        use the Lomb-Welch for averaging.

        :param frequency_grid: FrequencySamples object
        :return: FrequencySeries
        """
        return frequency_grid.lomb_scargle(self.times, self.value, norm=norm)

    def to_frequencyseries(self, transform: FourierTransform):
        if self.kind == 'complex':
            raise TypeError("transform only work with real timeSeries data")

        tmp = transform.backward(self, N=len(self))
        freqs = FrequencySamples(initial_array=transform.get_frequency(N=len(self)))
        return FrequencySeries(tmp, frequency_grid=freqs, epoch=self.epoch)

    def match(self, other, transform: FourierTransform, psd=None, tol=0.1):
        return self.to_frequencyseries(transform).match(other, psd=psd, tol=tol)

    def plot(self, axis=None, label="data", _show=False):
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)

        axis.plot(self.times, self.value, label=label)
        axis.set_title("TimeSeries values")
        axis.set_xlabel("times (sec)")
        plt.legend()
        if _show:
            plt.show()

    def add_data(self, point_data, point_time):
        self._data = np.append(self._data, point_data)
        self._times.add_point(point_time)


    # TODO: not used
    # def time_slice(self, start, end):
    #     start_idx = np.argmin(np.abs(self._times.offset - start))
    #     end_idx = np.argmin(np.abs(self._times.end - end))
    #     return self.slice_by_indexes(start_idx, end_idx), \
    #         self._times.slice_by_values(start, end)

