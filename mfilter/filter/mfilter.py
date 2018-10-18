
from mfilter.types import TimeSeries, FrequencySeries, FrequencySamples, TimesSamples
from mfilter.filter.matchedfilter import *
from mfilter.regressions.dictionaries import Dictionary
from mfilter.regressions.regressors import BasicRegression, SGDRegression
import abc


def sigmasq(htilde: FrequencySeries, psd=None):
    norm = 2*htilde.delta_f

    if psd is None:
        sq = htilde.inner()
    else:
        assert psd.delta_f == htilde.delta_f
        assert (psd.frequencies == htilde.frequencies).all()
        sq = htilde.weighted_inner(weight=psd)

    return sq.real * norm


def to_snr(time: TimesSamples, corr: FrequencySeries, reg: BasicRegression):
    reg.reset()
    reg.create_dict(time, corr.frequency_object)
    return corr.to_timeseries(method="regression", reg=reg)


def sigma(htilde: FrequencySeries, psd: FrequencySeries=None):
    return np.sqrt(sigmasq(htilde, psd=psd))


def correlation(stilde: FrequencySeries, htilde: FrequencySeries, psd: FrequencySeries=None):
    if psd is None:
        psd = 1
    return FrequencySeries(stilde * htilde.conj() / psd,
                           frequency_grid=htilde.frequency_object,
                           epoch=htilde.epoch)


def mfilter(time: TimesSamples, stilde: FrequencySeries,
            htilde: FrequencySeries, reg: BasicRegression, psd=None):

    norm = 2 / sigma(htilde, psd=None)
    corr = correlation(stilde, htilde, psd=psd)
    q = to_snr(time, corr, reg=reg)
    q *= norm
    return q


class SecuentialMFilter(object):
    def __init__(self, event_duration, upgrade_rate=1):
        self.event_duration = event_duration
        self.upgrade_rate = upgrade_rate
        self.data = None
        self.data_window = None
        self.freq = None
        self._reg = None
        self.coefs_ = None
        self.trigger_upgrade = False
        self.idx_start = 0
        self.idx_end = len(self.data)-1
        self.last_segment = False

    def set(self, data, frequency=None, **kwargs):
        self.data = self._validate_data(data, **kwargs)
        if self.data.times.min() > 0:
            raise ValueError("times must be centered at 0")
        self.freq = self._validate_freq(frequency, **kwargs)
        max_idx = np.argmin(np.abs(self.data.times - self.event_duration))
        self.data_window = TimeSeries(self.data[:max_idx+1],
                                      times=self.data.times[:max_idx+1])

        dictionary = Dictionary(self.data_window.times, self.freq)
        self._reg = SGDRegression(alpha=kwargs.get("alpha", 0.0001),
                                  max_iter=kwargs.get("max_iter", 500),
                                  tol=kwargs.get("tol", 0.01),
                                  phi=dictionary)
        self._idx = 0

    def add_observation(self, data_point, time_point):
        self.data.add_data(data_point, time_point)

    def _upgrade_window(self, points=1, **kwargs):
        self._idx += points
        aux = self.data[self._idx:]
        aux_time = self.data.times[self._idx:]
        aux_time -= min(aux_time)
        idx_max = np.argmin(np.abs(aux_time - self.event_duration))
        self.data_window = TimeSeries(aux[:idx_max],
                                      times=aux_time[:idx_max])
        dictionary = Dictionary(self.data_window.times, self.freq)
        self._reg = SGDRegression(alpha=kwargs.get("alpha", 0.0001),
                                  max_iter=kwargs.get("max_iter", 500),
                                  tol=kwargs.get("tol", 0.01),
                                  phi=dictionary)

    def _validate_data(self, data, **kwargs):
        if isinstance(data, TimeSeries):
            return data
        else:
            return TimeSeries(data, times=kwargs.get("times", None))

    def _validate_freq(self, frequency, **kwargs):
        if isinstance(frequency, FrequencySamples):
            return frequency
        else:
            return FrequencySamples(
                initial_array=frequency,
                input_time=kwargs.get("input_time", None),
                minimum_frequency=kwargs.get("minimum_frequency", None),
                maximum_frequency=kwargs.get("maximum_frequency", None),
                samples_per_peak=kwargs.get("samples_per_peak", None),
                nyquist_factor=kwargs.get("nyquist_factor", None),
                df=kwargs.get("df", None))

    def compute_templates(self, template_generator, params=None, N=500):
        if not isinstance(params, dict):
            raise ValueError("no params where given")

        times = TimesSamples(np.linspace(0, self.event_duration, N))
        dictionary = Dictionary(times, self.freq)
        reg = SGDRegression(alpha=10**(-10), tol=0.001, phi=dictionary)
        htildes = {}
        for k, v in params.items():
            temp = template_generator(times, v)
            htildes[k] = temp.to_frequencyseries(reg=reg)
            if reg.score(temp) < 0.8:
                print("warning: score to low for {}".format(k))

        return htildes

    def _get_idx(self, time_start):
        if time_start + self.event_duration > self.data.times.duration:
            self.last_segment = True
            idx_start = np.argmin(np.abs(self.data.times - self.data.times.max() + self.event_duration))
            idx_end = len(self.data) - 1
        else:
            idx_start = np.argmin(np.abs(self.data.times - time_start))
            idx_end = np.argmin(np.abs(self.data.times - time_start - self.event_duration))

        return slice(int(idx_start), idx_end+1)

    def _segment_match(self, time_start, htildes, overfit=0.5):
        if any(not isinstance(h, FrequencySeries)  for k, h in htildes.items()):
            raise ValueError("templates must be FrequencySeries")

        window_range = self._get_idx(time_start)

        window_time = TimesSamples(self.data.times[window_range])
        window_data = TimeSeries(self.data[window_range],
                                 times=window_time)
        n_seg = len(window_data)//10
        psd_window = self.freq.lomb_welch(window_time, window_data,
                                          n_seg, overfit)
        snr = {}
        for k, htilde in htildes.items():
            snr[k] = matched_filter(htilde, window_data, psd=psd_window,
                                    frequency_grid=self.freq, unitary_energy=True,
                                    reg=self._reg)

        self.idx_end = window_range.stop
        self.idx_start = window_range.start

        return snr

    def _segment_match_previous_template(self, window_range,
                                         htildes, overfit=0.5):

        window_time = TimesSamples(self.data.times[window_range])
        window_data = TimeSeries(self.data[window_range],
                                 times=window_time)
        n_seg = len(window_data) // 10
        psd_window = self.freq.lomb_welch(window_time, window_data,
                                          n_seg, overfit)
        snr = {}
        for k, htilde in htildes.items():
            snr[k] = matched_filter(htilde, window_data, psd=psd_window,
                                    frequency_grid=self.freq,
                                    unitary_energy=True,
                                    reg=self._reg)

        self.idx_end = window_range.stop
        self.idx_start = window_range.start

        return snr

    def _segment_match_calculating_template(self, window_range,
                                            templates_generator, overfit=0.5,
                                            **kwargs):

        window_time = TimesSamples(self.data.times[window_range])
        window_data = TimeSeries(self.data[window_range],
                                 times=window_time)
        templates = templates_generator(window_time, **kwargs)
        temp_dict = Dictionary(window_time, self.freq)
        temp_reg = SGDRegression(alpha=kwargs.get("alpha", 0.0001),
                                 max_iter=kwargs.get("max_iter", 500),
                                 tol=kwargs.get("tol", 0.001), phi=temp_dict)
        temp_reg.coef_ = self._reg.coef
        n_seg = len(window_data) // 10
        psd_window = self.freq.lomb_welch(window_time, window_data,
                                          n_seg, overfit)
        snr = {}
        for i in range(len(templates)):
            htilde = templates[i].to_frequencyseries(reg=temp_reg)
            snr[i] = matched_filter(htilde, window_data, psd=psd_window,
                                    frequency_grid=self.freq,
                                    unitary_energy=True,
                                    reg=self._reg)

        self.idx_end = window_range.stop
        self.idx_start = window_range.start

        return snr

    def match_all_same_htildes(self, htildes_generator, overfit=0.5,
                               **kwargs):
        time_start = self.data.times.min()
        self.last_segment = False
        htildes = htildes_generator(self.event_duration, self.freq, **kwargs)
        snrs = []
        while not self.last_segment:
            window_range = self._get_idx(time_start)
            snrs.append(self._segment_match_previous_template(window_range,
                                                              htildes,
                                                              overfit=overfit))
            time_start = time_start + self.event_duration

        return snrs

    def match_all_not_same_htildes(self, templates_generator, overfit=0.5,
                                   **kwargs):
        time_start = self.data.times.min()
        self.last_segment = False
        snrs = []
        while not self.last_segment:
            window_range = self._get_idx(time_start)
            snrs.append(self._segment_match_calculating_template(window_range,
                                                                 templates_generator,
                                                                 overfit=overfit,
                                                                 **kwargs))
            time_start = time_start + self.event_duration

        return snrs

    # def match_all(self, templates_generator, overfit=0.5, **kwargs):
    #     """
    #
    #     :param templates_generator:     array of templates generators
    #     :param kwargs:                  params for the template generator
    #     """
    #     time_start = self.data.times.min()
    #     self.last_segment = False
    #     while not self.last_segment:
    #         window_range = self._get_idx(time_start)
    #         window_time = TimesSamples(self.data.times[start_idx:end_idx])
    #         window = TimeSeries(self.data[start_idx:end_idx],
    #                             times=window_time)
    #         templates = templates_generator(window_time, **kwargs)
    #         seg_n = len(window_time)//10
    #         self._reg.set_dict(window_time, self.freq)
    #         psd = self.freq.lomb_welch(window_time, window, seg_n, overfit)
    #         stilde = window.to_frequencyseries(reg=self._reg)
    #         for temp in templates:
    #             snr = matched_filter(temp, stilde, psd=psd,
    #                                  frequency_grid=self.freq,
    #                                  unitary_energy=True, reg=self._reg)











