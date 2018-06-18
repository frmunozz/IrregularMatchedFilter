
"""Main Matching Filter Implementation"""

from .implementations.template import Templates
from .implementations.Segment import DataSegment
from .utils import Window
import logging
from nfft import nfft_adjoint, nfft
from mfilter.implementations.regressions import *
from .types.timeseries import TimeSeries
from .types.frequencyseries import FrequencySeries
import matplotlib.pyplot as plt
import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def get_frequency(times, samples_per_peak=5,
                  nyquist_factor=5, minimum_frequency=None,
                  maximum_frequency=None,
                  return_freq_limits=False):
    # pdb.set_trace()
    baseline = max(times) - min(times)
    n_samples = len(times)

    df = 1 / (baseline * samples_per_peak)

    if minimum_frequency is None:
        minimum_frequency = 0.5 * df

    if maximum_frequency is None:
        # bad estimation of nyquist limit
        average_nyq = 0.5 * n_samples / baseline
        # to fix this bad estimation, amplify the estimated value by 5
        maximum_frequency = nyquist_factor * average_nyq

    Nf = 1 + int(np.round((maximum_frequency - minimum_frequency) / df))

    if return_freq_limits:
        return minimum_frequency, minimum_frequency + df * (Nf - 1)
    else:
        return minimum_frequency + df * np.arange(Nf)


def cast_into_ft(coefs):
    n_freqs = int(len(coefs) / 2)
    ft = 1j * np.zeros(n_freqs)
    for i in range(n_freqs):
        ft[i] = coefs[i] + 1j * coefs[i + n_freqs]
    return ft


class BasicMatchedFilter(object):
    def __init__(self, data: TimeSeries, template: Array):
        if not isinstance(data, TimeSeries):
            raise ValueError("need to use a TimeSeries object as 'data'")
        if not isinstance(template, (TimeSeries, FrequencySeries)):
            raise ValueError("need to use a TimeSeries or FrequencySeries" +
                             " as 'template")

        self.data = data
        self.template = template
        self.same_size()

    def same_size(self):
        if isinstance(self.template, TimeSeries):
            gap = len(self.data) - len(self.template)
            if gap > 0:
                self.template.resize(len(self.data))
                self.template.times.replace(self.data.times)
            elif gap < 0:
                self.data.resize(len(self.template))
                self.data.times.resize(self.template.times)

            assert self.data == self.template

    def _ft(self, **kwargs):
        # dummy
        return 0, 0

    def _compute_corr(self, data_ft, temp_ft, psd):
        pass

    def _h_norm(self, temp_ft, psd):
        pass

    def _psd(self, **kwargs):
        pass

    def snr(self, **kwargs):
        # pdb.set_trace()
        psd = kwargs.get('psd', None)
        if psd is None:
            psd = self._psd(**kwargs)
        data_ft, temp_ft = self._ft(**kwargs)

        corr = self._compute_corr(data_ft, temp_ft, psd)
        #corr.plot()
        h_norm = self._h_norm(temp_ft, psd)
        print("h_norm is ", h_norm.real)
        norm = 4 * data_ft.df
        print("norm is ", norm)
        z = corr.reconstruct(self._reg, times=self.data.times)

        #z.plot()

        snr = norm * z.real / np.sqrt(h_norm.real * norm)

        return TimeSeries(snr, times=self.data.times)


class MatchedFilterRegression(BasicMatchedFilter):
    def __init__(self, data: TimeSeries, template: Array, reg: BasicRegression):
        super().__init__(data, template)
        self._reg = reg

    def _set_dictionary(self, **kwargs):
        freqs = kwargs.get("frequency", None)

        if isinstance(freqs, (list, np.ndarray)):
            freqs = Array(np.array(freqs), delta=freqs[1] - freqs[0])

        if freqs is None:
            if isinstance(self.template, FrequencySeries):
                logger.warning("using auto calculation of dictionary")
                freqs = self.template.frequency
            else:
                raise ValueError("need to have q valid frequency")

        self._reg.set_dict(self.data.times, freqs)

    def _data_ft(self):
        return FrequencySeries(self._reg.get_ft(self.data),
                               frequency=self._reg.frequency, df=self._reg.df)

    def _temp_ft(self):
        if isinstance(self.template, FrequencySeries):
            return self.template
        else:
            return FrequencySeries(self._reg.get_ft(self.template),
                                   frequency=self._reg.frequency,
                                   df=self._reg.df)

    def _ft(self, **kwargs):
        self._set_dictionary(**kwargs)
        return self._data_ft(), self._temp_ft()

    def _compute_corr(self, data_ft, temp_ft, psd):
        simple_corr = np.multiply(data_ft.data, temp_ft.data.conjugate())
        return FrequencySeries(simple_corr / psd, frequency=data_ft.frequency,
                               df=data_ft.df)

    def _h_norm(self, temp_ft, psd):
        # pdb.set_trace()
        simple_corr = np.multiply(temp_ft.data, temp_ft.data.conjugate())
        return (simple_corr / psd).sum()

    def _psd(self, **kwargs):
        raise ValueError("estimation of noise PSD not implemented yet")
        pass


# TODO: not implemented yet
# class SegmentMatchedFilter:
#     def __init__(self, segm_time, segm_data, segm_template, noise_psd=None):
#         self.segm_time = segm_time
#         self.T = max(segm_time) - min(segm_time)
#         self.segm_data = segm_data
#         self.segm_temp = segm_template
#         self.noise_psd = self._validate_psd(noise_psd)
#
#     def _validate_psd(self, noise_psd):
#         if not isinstance(noise_psd, np.ndarray):
#             raise ValueError("for now you must give to the match filter" +
#                              " a one sided power spectrum noise previously," +
#                              " in the future we expect to implement an" +
#                              " estimator for the PSD")
#         return noise_psd
#
#     def compute_snr(self, method="nfft", normalize="noise",
#                     regression_method="elasticnet", prm_regresion=None,
#                     frequencies=None):
#         noise = None
#         if normalize is "noise":
#             # simulate a situation where the signal is only noise, for this
#             # we use a rustic approximation where whe generate a gaussian
#             # noise of 0 mean and 0.5 * max(signal) deviation
#             noise = np.random.normal(0, 0.5 * max(self.segm_data),
#                                      len(self.segm_data))
#         F = None
#         if method is "nfft":
#             df = 1 / self.T
#             data_ft, temp_ft, noise_ft, pw = self._use_nfft()
#             corr = 4 * df * data_ft * temp_ft.conjugate() / pw
#             z = nfft(self.segm_time, corr)
#
#         elif method is "regression":
#             k, df = self._frequencies(frequencies)
#             F = np.exp(2j * np.pi * self.segm_time.reshape(-1, 1) * k)
#             Phi = np.hstack((F.real, F.imag))
#
#             data_ft, temp_ft, noise_ft = self._use_regression(Phi,
#                                                               regression_method,
#                                                               prm_regresion,
#                                                               noise=noise)
#             pw = self.noise_psd
#             corr = 4 * df * data_ft * temp_ft.conjugate() / pw
#             z = np.dot(F, corr)
#
#         else:
#             raise ValueError("method '{}' not implemented".format(method))
#
#         if normalize is "noise":
#             # use the ft of the estimated noise to calculate values of SNR for
#             # a signal of only noise
#             sigma = 4 * df * np.mean(noise_ft * temp_ft.conjugate() / pw)
#
#         elif normalize is "off":
#             sigma = (4 * df)**2
#
#         elif normalize is "template":
#             sigma = 4 * df * (temp_ft * temp_ft.conjugate() / pw).sum()
#
#         else:
#             raise ValueError("normalize by '{}' not".format(normalize) +
#                              " implemented")
#
#         return np.abs(z) / np.sqrt(sigma)
#
#     def _frequencies(self, frequencies):
#         if frequencies is None:
#             frequencies = get_frequency(self.segm_time)
#
#         if len(frequencies) < len(self.segm_time):
#             logger.warning("using under complete dictionary")
#
#         if max(frequencies) * 2 > len(self.segm_time) / self.T:
#             logger.warning("Nyquist-Shannon sampling theorem not satisfied")
#
#         return frequencies, frequencies[1] - frequencies[0]
#
#     def _use_regression(self, F, regression_method, prm_regresion, noise=None):
#         if prm_regresion is None:
#             prm_regresion = {}
#
#         if "nfold" in prm_regresion:
#             n_fold = prm_regresion["nfold"]
#         else:
#             n_fold = 3
#
#         if regression_method is "elasticnet":
#             if "alphas" in prm_regresion:
#                 alphas = prm_regresion["alphas"]
#             else:
#                 alphas = np.logspace(-5, 0, num=10, base=10)
#
#             if "l1ratio" in prm_regresion:
#                 l1_ratio = prm_regresion["l1ratio"]
#             else:
#                 l1_ratio = np.linspace(0.1, 1, 10)
#             reg = linear_model.ElasticNetCV(l1_ratio=l1_ratio, alphas=alphas,
#                                             cv=n_fold)
#
#         elif regression_method is "ridge":
#             if "alphas" in prm_regresion:
#                 alphas = prm_regresion["alphas"]
#             else:
#                 alphas = np.logspace(-5, 0, num=10, base=10)
#             reg = linear_model.RidgeCV(alphas=alphas, cv=n_fold)
#
#         elif regression_method is "omp":
#             reg = linear_model.OrthogonalMatchingPursuitCV(cv=n_fold)
#
#         else:
#             raise ValueError("regression method {}".format(regression_method) +
#                              " not implemented")
#         reg.fit(F, self.segm_data)
#         data_ft = cast_into_ft(reg.coef_)
#         reg.fit(F, self.segm_temp)
#         temp_ft = cast_into_ft(reg.coef_)
#         if noise is None:
#             return data_ft, temp_ft, None
#         else:
#             reg.fit(F, noise)
#             noise_ft = cast_into_ft(reg.coef_)
#             return data_ft, temp_ft, noise_ft
#
#     def _use_nfft(self, noise=None):
#
#         # duplicate the one sided pw to cover the negative frequencies,
#         # for this we supuse real data/temp to has a symmetry in frequencies
#         pw = np.append(self.noise_psd, self.noise_psd)
#         if len(self.segm_data) % 2 == 0:
#             # then delete 2 frequencies to match the ft
#             pw = np.delete(pw, len(pw)-1)
#             pw = np.delete(pw, len(pw)-1)
#
#         data_ft = self._compute_nfft(self.segm_data)
#         temp_ft = self._compute_nfft(self.segm_temp)
#         assert len(pw) == len(data_ft)
#
#         if noise is None:
#             return data_ft, temp_ft, None, pw
#         else:
#             noise_ft = self._compute_nfft(noise)
#             return data_ft, temp_ft, noise_ft, pw
#
#     def _compute_nfft(self, y):
#         ft = nfft_adjoint(self.segm_time, y, len(y))
#
#         # remove 0 freq which is the just half
#         ft = np.delete(ft, len(self.segm_time) // 2)
#
#         # to get this as even remove another freq, we choose the last one
#         ft = np.delete(ft, len(ft)-1)
#         return ft
#
#
# def _validate_inputs(t, y, dy):
#     # Validate shapes of inputs
#     if dy is None:
#         t, y = np.broadcast_arrays(t, y, subok=True)
#     else:
#         t, y, dy = np.broadcast_arrays(t, y, dy, subok=True)
#     if t.ndim != 1:
#         raise ValueError("Inputs (t, y, dy) must be 1-dimensional")
#     return t, y, dy
#
#
# def _set_window(window, prm):
#     if window is None:
#         window = Window(window, prm)
#     elif isinstance(window, str):
#         window = Window(window, prm)
#
#     elif not isinstance(window, Window):
#         raise ValueError("window type ({}) not supported".format(type(window)))
#
#     return window
#
#
# def _get_template(template, prm_temp):
#     if template is None:
#         # initialize a 'one_sin' template by default
#         template = Templates(template_type=None)
#
#     elif isinstance(template, str):
#         if isinstance(prm_temp, dict):
#             # use a template from the template implementation
#             template = Templates(template_type=template, prm=prm_temp)
#         else:
#             raise ValueError("if template is a string, then input prm_temp " +
#                              "must be of type 'dict' " +
#                              "instead of {}".format(type(prm_temp)))
#
#     elif isinstance(template, np.ndarray):
#         if not (isinstance(prm_temp, np.ndarray)
#                 and template.shape != prm_temp.shape):
#             raise ValueError("if 'template' is an array, then " +
#                              "'prm_temp' should be also an array " +
#                              "and both must have shame shape")
#
#     elif not isinstance(template, Templates):
#         raise ValueError("unexpected type {} ".format(type(template)) +
#                          " for input 'template' should be one of: " +
#                          "{'str', 'np.ndarray', 'Templates', 'NoneType}")
#
#     return template, prm_temp
#
#
# def _validate_segment(segment, prm_segment, t, data):
#
#     t_segment = None
#     ovlp_fact = None
#
#     if isinstance(segment, np.ndarray):
#         t_segment = segment[1] - segment[0]
#         if isinstance(prm_segment, dict) and 'ovlp_fact' in prm_segment:
#             ovlp_fact = prm_segment['ovlp_fact']
#
#     elif segment is None:
#         if isinstance(prm_segment, dict):
#             if 'ovlp_fact' in prm_segment:
#                 ovlp_fact = prm_segment['ovlp_fact']
#             if 't_segment' in prm_segment:
#                 t_segment = prm_segment['t_segment']
#     elif not isinstance(segment, DataSegment):
#         raise ValueError("input segment type" +
#                          "({})".format(type(segment)) +
#                          "unexpected, should be (DataSegment " +
#                          "or np.ndarray or NoneType)")
#
#     if not isinstance(segment, DataSegment):
#         segment = DataSegment(t, data, t_segment=t_segment,
#                               overlap_factor=ovlp_fact)
#
#     if prm_segment is None:
#         # compute the next block in the counter
#         segment_n = segment.set_segment_number()
#         prm_segment = {"segment_n": segment_n}
#     elif isinstance(prm_segment, dict):
#         if "segment_n" in prm_segment:
#             segment_n = prm_segment["segment_n"]
#         else:
#             segment_n = None
#
#         segment_n = segment.set_segment_number(segment_n=segment_n)
#         prm_segment["segment_n"] = segment_n
#     else:
#         raise ValueError("input prm_segment type" +
#                          "({})".format(type(prm_segment)) +
#                          "unexpected, should be (dict)")
#
#     return segment, prm_segment
#
# class MatchedFilter:
#     """Compute the Matched filter of some data with a template.
#
#     This implementations here are bases on code presented in [1]_, [2]_,
#     [3]_, [4]_ and [5]_
#
#     Parameters
#     ----------
#     :param t:           array_like
#                         observation times
#
#     :param y:           array_like
#                         observations associated with time t
#
#     :param dy:          float, array_like or Qauntity (optional)
#                         error or sequence of observacional error associated
#                         with times t, it's used for the Lomb-Scargle
#                         periodogram, by default use 1.
#
#     :param segment:     DataSegment or a np.ndarray or None
#                         the segment of the data to use for the
#                         computation of the Matching Filter.
#
#     :param prm_segment: np.ndarray or None (optional)
#                         if segment is a DataSegment, then use
#                         this params to compute the segment.
#
#     :param template:    array_like or string or Template ir None (optional)
#                         the template to used for the match filter, could be
#                         an array, an instance of Templates or None (default
#                         template)
#
#     :param prm_temp:    dict or array_like or None (optional)
#                         if input 'template' not None then use this params
#                         to compute the template.
#
#     :param window:      array_like or string {'tukey', 'linear'}, optional
#                         window used to remove effect at the beginning and end
#                         of the data stretch
#
#     :param prm_window:  array_like (optional)
#                         parameters used for calculate the specified window.
#
#
#     References
#     ----------
#     .. [1] Vanderplas, J., Connolly, A. Ivezic, Z. & Gray, A. *Introduction to
#         astroML: Machine learning for astrophysics*. Proceedings of the
#         Conference on Intelligent Data Understanding (2012)
#     .. [2] VanderPlas, J. & Ivezic, Z. *Periodograms for Multiband Astronomical
#         Time Series*. ApJ 812.1:18 (2015)
#     .. [3] Allen, B., Anderson,  W. G., Brady, P. R., Brown, D. A.,
#         & Creighton, J. D. E. *FINDCHIRP: An algorithm for detection of
#         gravitational waves from inspiraling compact binaries*.
#         Phys.Rev. D85, 122006 (2012)
#     .. [4] VanderPlas, J. *Understanding the Lomb-Scargle Periodogram*.
#         ArXiv e-prints, [1703.09824], mar, (2017)
#     .. [5] Nitz, A. H., Harry, I. W., Willis, J. L., Biwer, C. M.,
#         Brown, D. A., Pekowsky, L. P., Dal Canton, T., Williamson, A. R.,
#         Dent, T., Capano, C. D., Massinger, T. T., Lenon, A. K., Nielsen, A.,
#         & Cabero, M. *PyCBC Software* github.com/ligo-cbc/pycbc (2016)
#     """
#     def __init__(self, t, y, dy=None, segment=None, prm_segment=None,
#                  template=None, prm_temp=None,
#                  window=None, prm_window=None):
#         self.t, self.y, self.dy = _validate_inputs(t, y, dy)
#         self.segm, self.prm_segm = _validate_segment(segment,
#                                                      prm_segment,
#                                                      t, y)
#         self.window = _set_window(window, prm_window)
#         self.temp, self.prm_temp = _get_template(template, prm_temp)
#         self.stop = False
#
#     def _compute_template(self, segment_times, epsilon=0.1):
#         if isinstance(self.temp, Templates):
#             return self.temp.compute_template(segment_times)
#         elif isinstance(self.temp, np.ndarray):
#             if max(self.prm_temp) - max(segment_times) > self.t * epsilon:
#                 raise ValueError("difference between times measures by data" +
#                                  " and template shouldn't be higher than " +
#                                  "{} units".format(epsilon * self.t))
#             return self.temp
#
#
#     def _next_segment(self):
#         segment_num = self.prm_segm["segment_n"]
#         if segment_num == self.segm.n_segment - 1:
#             print("stop at the last segment, which is: "
#                   + str(self.prm_segm["segment_n"]))
#             self.stop = True
#         else:
#             self.prm_segm["segment_n"] = segment_num + 1
#
#     def _estimate_psd(self, noise_psd):
#         if noise_psd is None:
#             raise ValueError("estimation of noise psd not implemented yet")
#         return noise_psd
#
#     def select_segment(self, noise_psd=None,
#                        segment_num: int = None, epsilon=0.1):
#         segm_time, segm_data, segm_temp = self._segmentation(segment_num,
#                                                              epsilon)
#
#         return SegmentMatchedFilter(segm_time, segm_data, segm_temp,
#                                     noise_psd=self._estimate_psd(noise_psd))
#
#     def snr(self, noise_psd=None, method="nfft", normalize="template",
#             regression_method="ridge",
#             prm_regresion=None, segment_num:int = None,
#             minimum_frequency: float = None,
#             maximum_frequency: float = None,
#             nyquist_factor: int = 5,
#             samples_per_peak: int = 5, epsilon=0.1):
#
#         mf_object = self.select_segment(noise_psd=noise_psd,
#                                         segment_num=segment_num,
#                                         epsilon=epsilon)
#
#         frequencies = get_frequency(mf_object.segm_time,
#                                     samples_per_peak=samples_per_peak,
#                                     nyquist_factor=nyquist_factor,
#                                     minimum_frequency=minimum_frequency,
#                                     maximum_frequency=maximum_frequency,
#                                     return_freq_limits=False)
#
#         return mf_object.compute_snr(method=method, normalize=normalize,
#                                      regression_method=regression_method,
#                                      prm_regresion=prm_regresion,
#                                      frequencies=frequencies)
#
#     def _segmentation(self, segment_num, epsilon):
#         if self.stop and segment_num is None:
#             logger.warning("you have reached the last segment")
#             return None, None, None
#
#         if segment_num is None:
#             segment_num = self.prm_segm["segment_n"]
#         segment_times, \
#         segment_data = self.segm.compute_segment(segment_number=segment_num)
#
#         valid_template = self._compute_template(segment_times, epsilon=epsilon)
#         self._next_segment()
#         return segment_times, segment_data, valid_template
#
