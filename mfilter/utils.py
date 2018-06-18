import numpy as np
from scipy import signal


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




class Window:
    def __init__(self, window_type: str=None, prm: np.ndarray=None):
        self.window_type, self.prm = self._set_window(window_type, prm=prm)

    def _set_window(self, window_type: str, prm: np.ndarray=None):
        if window_type is None:
            window_type = 'tukey'

        if window_type == 'tukey':
            alpha = 1/8 if prm is None else prm[0]
            prm = [alpha]

        elif window_type == 'linear':
            prm = 1 if prm is None else prm[0]

        else:
            raise ValueError("for now there is only implemented a tukey " +
                             "window and a linear window")
        return window_type, prm

    def compute(self, size: int):
        window = None
        if self.window_type == 'tukey':
            window = signal.tukey(size, alpha=self.prm[0])

        elif self.window_type == 'linear':
            window = np.ones(size) * self.prm[0]
        return window
