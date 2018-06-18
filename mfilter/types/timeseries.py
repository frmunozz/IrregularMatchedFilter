import numpy as np
from mfilter.implementations.regressions import *
from mfilter.types.arrays import Array
from mfilter.types.frequencyseries import FrequencySamples
import matplotlib.pyplot as plt


class TimesSamples(Array):
    def __init__(self, n, delta=None, regular=False, struct="slight",
                 clear=True, **kwargs):
        if delta is None:
            raise ValueError("need to receive a valid delta of times")

        if regular:
            offset = kwargs.get("offset", 0)
            arr = offset + np.arange(n) * delta
        else:
            arr = IrregularTimeSamples(n, delta, **kwargs)

        super().__init__(arr.compute(struct=struct, clear=clear, **kwargs),
                         delta=delta)

    @property
    def times(self):
        return self.data

    @property
    def dt(self):
        return self.delta

    @property
    def fs(self):
        return len(self) / self.duration


class IrregularTimeSamples(object):
    def __init__(self, n: int, dt: float, grid: np.ndarray=None):
        if dt <= 0:
            raise ValueError("'dt' need to be positive and non zero")

        if n != len(grid):
            raise ValueError("the 'grid' used need to be of length 'n'")

        self.n = n
        self.dt = dt
        self.t = grid if grid is not None else np.arange(n) * dt

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
        gamma = np.random.choice([0.5, 1, 2, 3])
        offset = np.random.uniform(0, self.n * self.dt)
        empty_window = np.random.uniform(0, self.n * self.dt * 0.5)

        self.t[2 * (self.n // 5):4 * (self.n // 5)] *= gamma
        self.t[3 * (self.n // 5):] += empty_window
        self._add_irregularities()
        self._normalize(offset=offset)


class TimeSeries(Array):
    def __init__(self, initial_values, times=None, dt=None, regular=False,
                 **kwargs):
        super().__init__(initial_values)
        self._regular = regular
        self._times = self._validate_times(times, dt, len(self), **kwargs)

    def __eq__(self, other):
        """
        two time series are going to be equal when they have exactly same
        sampling times, the values of every time doesnt need to be the same
        :param other: the other time series
        """
        return self.times == other.times

    @property
    def times(self) -> Array:
        return self._times

    def _validate_times(self, times, dt, n, **kwargs):
        _boolean_aux = dt is not None and n is not None
        if times is None and _boolean_aux:
            # compute the times
            times = TimesSamples(n, delta=dt,
                                 regular=kwargs.get("regular", False),
                                 struct=kwargs.get("struct", "slight"),
                                 clear=kwargs.get("clear", True), **kwargs)

        elif isinstance(times, (list, np.ndarray)):
            dt = dt if self._regular else -1
            times = Array(times, delta=dt)

        return times

    def time_slice(self, start, end):
        start_idx = np.argmin(np.abs(self._times.offset - start))
        end_idx = np.argmin(np.abs(self._times.end - end))
        return self.slice_by_indexes(start_idx, end_idx), \
            self._times.slice_by_values(start, end)

    def auto_regression(self, regressor,
                        min_freq=None, max_freq=None,
                        samples_per_peak=5, nyquist_factor=2,
                        n_freqs=None):

        frequencies = FrequencySamples(self._times, n=n_freqs,
                                       minimum_frequency=min_freq,
                                       maximum_frequency=max_freq,
                                       samples_per_peak=samples_per_peak,
                                       nyquist_factor=nyquist_factor)

        return frequencies, self.regression(regressor, frequencies)

    def regression(self, regressor, frequencies: Array):
        new_dict = Dictionary(self._times.data, frequencies.data)
        return regressor.fit(self._data, phi=new_dict)

    def nfft(self):
        pass

    def psd_noise(self):
        pass

    def plot(self):
        plt.figure()
        plt.plot(self.times.data, self.data.real)
        plt.show()