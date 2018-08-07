
"""Main Matching Filter Implementation"""

import numpy as np
from astropy.stats import LombScargle
from .implementations.template import Templates
from .implementations.Segment import DataSegment
from .utils import Window
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def _validate_inputs(t, y, dy):
    # Validate shapes of inputs
    if dy is None:
        t, y = np.broadcast_arrays(t, y, subok=True)
    else:
        t, y, dy = np.broadcast_arrays(t, y, dy, subok=True)
    if t.ndim != 1:
        raise ValueError("Inputs (t, y, dy) must be 1-dimensional")
    return t, y, dy


def _set_window(window, prm):
    if window is None:
        return
    elif isinstance(window, str):
        window = Window(window, prm)

    return window


def _get_template(template, prm_temp):
    if template is None:
        # initialize a 'one_sin' template by default
        template = Templates(template_type=None)

    elif isinstance(template, str):
        if isinstance(prm_temp, dict):
            # use a template from the template implementation
            template = Templates(template_type=template, prm=prm_temp)
        else:
            raise ValueError("if template is a string, then input prm_temp " +
                             "must be of type 'dict' " +
                             "instead of {}".format(type(prm_temp)))

    elif isinstance(template, np.ndarray):
        if not (isinstance(prm_temp, np.ndarray)
                and template.shape != prm_temp.shape):
            raise ValueError("if 'template' is an array, then " +
                             "'prm_temp' should be also an array " +
                             "and both must have shame shape")

    elif not isinstance(template, Templates):
        raise ValueError("unexpected type {} ".format(type(template)) +
                         " for input 'template' should be one of: " +
                         "{'str', 'np.ndarray', 'Templates', 'NoneType}")

    return template, prm_temp


class SegmentMatchedFilter:
    def __init__(self, segment_num, times, data, window=None,
                 prm_segment=None, prm_window=None):
        self.segm, self.prm_Segm = self._validate_segment(segment_num,
                                                              prm_segment,
                                                              times, data)
        self.window = _set_window(window, prm_window)
        self.t, self.d, self.temp = self._compute_segments(times, data)

    def _compute_segments(self, times, data):
        return None, None, None

    def _validate_segment(self, segment, prm_segment, t, data):

        t_segment = None
        ovlp_fact = None

        if isinstance(segment, np.ndarray):
            t_segment = segment[1] - segment[0]
            if isinstance(prm_segment, dict) and 'ovlp_fact' in prm_segment:
                ovlp_fact = prm_segment['ovlp_fact']

        elif segment is None:
            if isinstance(prm_segment, dict):
                if 'ovlp_fact' in prm_segment:
                    ovlp_fact = prm_segment['ovlp_fact']
                if 't_segment' in prm_segment:
                    t_segment = prm_segment['t_segment']
        elif not isinstance(segment, DataSegment):
            raise ValueError("input segment type" +
                             "({})".format(type(segment)) +
                             "unexpected, should be (DataSegment " +
                             "or np.ndarray or NoneType)")

        if not isinstance(segment, DataSegment):
            segment = DataSegment(t, data, t_segment=t_segment,
                                  overlap_factor=ovlp_fact)

        if prm_segment is None:
            # compute the next block in the counter
            segment_n = segment.set_segment_number()
            prm_segment = {"segment_n": segment_n}
        elif isinstance(prm_segment, dict):
            if "segment_n" in prm_segment:
                segment_n = prm_segment["segment_n"]
            else:
                segment_n = None

            segment_n = segment.set_segment_number(segment_n=segment_n)
            prm_segment["segment_n"] = segment_n
        else:
            raise ValueError("input prm_segment type" +
                             "({})".format(type(prm_segment)) +
                             "unexpected, should be (dict)")

        return segment, prm_segment

class MatchedFilter:
    """Compute the Matched filter of some data with a template.

    This implementations here are bases on code presented in [1]_, [2]_,
    [3]_, [4]_ and [5]_

    Parameters
    ----------
    :param t:           array_like
                        observation times

    :param y:           array_like
                        observations associated with time t

    :param dy:          float, array_like or Qauntity (optional)
                        error or sequence of observacional error associated
                        with times t, it's used for the Lomb-Scargle
                        periodogram, by default use 1.

    :param segment:     DataSegment or a np.ndarray or None
                        the segment of the data to use for the
                        computation of the Matching Filter.

    :param prm_segment: np.ndarray or None (optional)
                        if segment is a DataSegment, then use
                        this params to compute the segment.

    :param template:    array_like or string or Template ir None (optional)
                        the template to used for the match filter, could be
                        an array, an instance of Templates or None (default
                        template)

    :param prm_temp:    dict or array_like or None (optional)
                        if input 'template' not None then use this params
                        to compute the template.

    :param window:      array_like or string {'tukey', 'linear'}, optional
                        window used to remove effect at the beginning and end
                        of the data stretch

    :param prm_window:  array_like (optional)
                        parameters used for calculate the specified window.


    References
    ----------
    .. [1] Vanderplas, J., Connolly, A. Ivezic, Z. & Gray, A. *Introduction to
        astroML: Machine learning for astrophysics*. Proceedings of the
        Conference on Intelligent Data Understanding (2012)
    .. [2] VanderPlas, J. & Ivezic, Z. *Periodograms for Multiband Astronomical
        Time Series*. ApJ 812.1:18 (2015)
    .. [3] Allen, B., Anderson,  W. G., Brady, P. R., Brown, D. A.,
        & Creighton, J. D. E. *FINDCHIRP: An algorithm for detection of
        gravitational waves from inspiraling compact binaries*.
        Phys.Rev. D85, 122006 (2012)
    .. [4] VanderPlas, J. *Understanding the Lomb-Scargle Periodogram*.
        ArXiv e-prints, [1703.09824], mar, (2017)
    .. [5] Nitz, A. H., Harry, I. W., Willis, J. L., Biwer, C. M.,
        Brown, D. A., Pekowsky, L. P., Dal Canton, T., Williamson, A. R.,
        Dent, T., Capano, C. D., Massinger, T. T., Lenon, A. K., Nielsen, A.,
        & Cabero, M. *PyCBC Software* github.com/ligo-cbc/pycbc (2016)
    """
    def __init__(self, t, y, dy=None, segment=None, prm_segment=None,
                 template=None, prm_temp=None,
                 window=None, prm_window=None):
        self.t, self.y, self.dy = _validate_inputs(t, y, dy)
        self.segment, self.prm_segment = self._validate_segment(segment,
                                                                prm_segment)
        self.window = _set_window(window, prm_window)
        self.temp, self.prm_temp = _get_template(template, prm_temp)
        self.stop = False

    def _validate_segment(self, segment, prm_segment):

        t_segment = None
        ovlp_fact = None

        if isinstance(segment, np.ndarray):
            t_segment = segment[1] - segment[0]
            if isinstance(prm_segment, dict) and 'ovlp_fact' in prm_segment:
                ovlp_fact = prm_segment['ovlp_fact']

        elif segment is None:
            if isinstance(prm_segment, dict):
                if 'ovlp_fact' in prm_segment:
                    ovlp_fact = prm_segment['ovlp_fact']
                if 't_segment' in prm_segment:
                    t_segment = prm_segment['t_segment']
        elif not isinstance(segment, DataSegment):
            raise ValueError("input segment type" +
                             "({})".format(type(segment)) +
                             "unexpected, should be (DataSegment " +
                             "or np.ndarray or NoneType)")

        if not isinstance(segment, DataSegment):
            segment = DataSegment(self.t, self.y, t_segment=t_segment,
                                  overlap_factor=ovlp_fact)

        if prm_segment is None:
            # compute the next block in the counter
            segment_n = segment.set_segment_number()
            prm_segment = {"segment_n": segment_n}
        elif isinstance(prm_segment, dict):
            if "segment_n" in prm_segment:
                segment_n = prm_segment["segment_n"]
            else:
                segment_n = None

            segment_n = segment.set_segment_number(segment_n=segment_n)
            prm_segment["segment_n"] = segment_n
        else:
            raise ValueError("input prm_segment type" +
                             "({})".format(type(prm_segment)) +
                             "unexpected, should be (dict)")

        return segment, prm_segment

    def _compute_template(self, segment_times, epsilon=0.1):
        if isinstance(self.temp, Templates):
            return self.temp.compute_template(segment_times)
        elif isinstance(self.temp, np.ndarray):
            if max(self.prm_temp) - max(segment_times) > self.t * epsilon:
                raise ValueError("difference between times measures by data" +
                                 " and template shouldn't be higher than " +
                                 "{} units".format(epsilon * self.t))
            return self.temp


    def _next_segment(self):
        segment_num = self.prm_segment["segment_n"]
        if segment_num == self.segment.n_segment - 1:
            print("stop at the last segment, which is: "
                  + str(self.prm_segment["segment_n"]))
            self.stop = True
        else:
            self.prm_segment["segment_n"] = segment_num + 1

    def autofrequency(self, samples_per_peak=5,
                      nyquist_factor=5, minimum_frequency=None,
                      maximum_frequency=None,
                      return_freq_limits=False):

        baseline = max(self.t) - min(self.t)
        n_samples = len(self.t)

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

    # def get_nfft(self, times, data, template, frequencies):

    def compute_nfft(self, segment_num: int = None,
                        minimum_frequency: float = None,
                        maximum_frequency: float = None,
                        nyquist_factor: int = 5,
                        samples_per_peak: int = 5,
                        epsilon=0.1):
        """
        compute the non uniform fast fourier transform for the specific range
        of data with the specific window.

        :param minimum_frequency:   float (optional)
                                    If specified, use this minimum frequency
                                    rather than one chosen on the size of the
                                    data segment
        :param maximum_frequency:   float (optional)
                                    if specified, use this maximum frequency
                                    rather than one chosen based on the average
                                    nyquist frequency
        :param nyquist_factor:      integer or float (optional)
                                    the multiple of the average nyquist
                                    frequency used to choose the maximum
                                    frequency if maximum_frequency is not
                                    provided
        :param samples_per_peak:    integer (optional)
        """
        segm_time, segm_data, segm_temp = self._segmentation(segment_num,
                                                             epsilon)

        frequencies = self.autofrequency(samples_per_peak=samples_per_peak,
                                         nyquist_factor=nyquist_factor,
                                         minimum_frequency=minimum_frequency,
                                         maximum_frequency=maximum_frequency,
                                         return_freq_limits=False)

        # dwindow = self.window.compute(len(segment_times))

        data_nfft, temp_nfft = self.get_nfft()

        power_noise = self.estimate_power_noise()

        return data_nfft, temp_nfft, power_noise

    def _segmentation(self, segment_num, epsilon):
        if self.stop and segment_num is None:
            logger.warning("you have reached the last segment")
            return None, None, None

        if segment_num is None:
            segment_num = self.prm_segment["segment_n"]
        segment_times, \
        segment_data = self.segment.compute_segment(segment_number=segment_num)

        valid_template = self._compute_template(segment_times, epsilon=epsilon)
        self._next_segment()
        return segment_times, segment_data, valid_template

    def _compute_dictionary(self, times: np.ndarray, freuencies: np.ndarray):
        return np.exp(2j * np.pi * times.reshape(-1, 1) * freuencies)

    def decompose_signal(self, frequencies: np.ndarray = None,
                         segment_num: int = None,
                         minimum_frequency: float = None,
                         maximum_frequency: float = None,
                         nyquist_factor: int = 5,
                         samples_per_peak: int = 5,
                         epsilon=0.1):

        segm_time, segm_data, segm_temp = self._segmentation(segment_num,
                                                             epsilon)
        if frequencies is None:
            frequencies = self.autofrequency(samples_per_peak=samples_per_peak,
                                             nyquist_factor=nyquist_factor,
                                             minimum_frequency=minimum_frequency,
                                             maximum_frequency=maximum_frequency,
                                             return_freq_limits=False)

        Phi = self._compute_dictionary(segm_time, frequencies)