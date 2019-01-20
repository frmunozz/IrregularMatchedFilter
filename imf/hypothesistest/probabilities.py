import numpy as np
from scipy.special import erfc, erfcinv


class HypothesisTesting(object):
    def __init__(self, sigma_square, false_alarm=None, p_detect=None):
        """
        Using hypothesis testing, which take the two hypothesis:


                        H0:  x(t) = n(t)
                        H1:  x(t) = n(t) + h(t)

        here H0 says that input data is noise with absent signal and H1 says
        that input data is noise with precense of signal which is the optimal
        desired signal h(t). This is a different approach to get the theory
        of linear filter. Here we will use the definition of likelihood ratio:

                                 if <
                             H0 satisfied
        ln[\Lambda] = (x|h)  ------------  ln[P(H0)/P(H1)] + (h|h)/2 = \eta
                                 if >
                             H1 satisfied

        where (x|h) and (h|h) are the linear filter convolution. Using this,
        for a given threshold \eta, we can say if our detected SNR which is
        just:

                    SNR = max[ (x|h)_t0 / \sqrt{(h|h)_0} ]

        correspond to a detection or not, but first we need to define this
        threshold, for this we can estimate his value due to given constrains.
        Here we use the NP criteria which consist of choose \eta such that
        maximize a probability of detection given by:


            P_D = \int_{\eta}^{\infty} f(x|H1) dx

        where F(x|H1) is the PDF of the input data in the case of hypothesis
        H1. And while maximize this P_D need to satisfy the false alarm rate
        constrains

            P_{FA} = \int_{\eta}^{\infty} f(x|H1) dx

        where f(x|H1) is a conditional PDF due to H1 being correct.


        :param sigma_square:        number
                                    result of product (h|h)
        :param false_alarm:         float
                                    probability of false alarm
        :param p_detect:            float
                                    probability of detect a signal
        """
        self._mu = np.sqrt(sigma_square)
        self._fa = false_alarm
        self._pd = p_detect
        self._threshold = None

    def _validate_input(self, snr):
        """
        for the input SNR, if is an array, then take the maximum value
        :param snr: SNR value(s)
        :return: best (max) value of the SNR value(s)
        """
        if isinstance(snr, (list, np.ndarray)):
            snr = np.max(snr)

        return snr

    def false_alarm(self, threshold=None):
        """
        estimate a false alarm probability from a given threshold, considering
        as true hypothesis H0 and distribution N(0, 1)
        :param threshold:
        :param fixed_value:
        :return:
        """
        if threshold is None:
            threshold = self._threshold

        if threshold is None:
            return self._fa

        if threshold < 0:
            raise ValueError("threshold cannot be negative")

        return erfc(threshold / np.sqrt(2)) / 2

    def p_detection(self, sigma_square=None, threshold=None):

        if threshold is None:
            threshold = self._threshold

        if threshold is None:
            threshold = np.sqrt(2) * erfcinv(self._fa * 2)

        if sigma_square is None:
            mu = self._mu
        else:
            mu = np.sqrt(sigma_square)

        return erfc((threshold - mu) / np.sqrt(2)) / 2

    def _threshold_from_p_detect(self):
        if self._pd is not None:
            self._threshold = np.sqrt(2) * erfcinv(self._pd * 2) + self._mu

    def _threshold_from_false_alarm(self):
        if self._fa is not None:
            self._threshold = np.sqrt(2) * erfcinv(self._fa * 2)

    def set_threshold(self, false_alarm=None, p_detect=None):
        if false_alarm is not None:
            self._fa = false_alarm

        if p_detect is not None:
            self._pd = p_detect

        self._threshold = None
        self._threshold_from_false_alarm()
        if self._threshold is None:
            self._threshold_from_p_detect()
        return self._threshold

    def decide(self, snr_max, binary_return=True):
        if self._threshold is None:
            raise ValueError("must to estimate threshold first")

        if binary_return:
            return 0 if snr_max < self._threshold else 1
        else:
            return snr_max < self._threshold

    @property
    def threshold(self):
        return self._threshold