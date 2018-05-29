import numpy as np
from scipy import signal


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
