import numpy as np
import matplotlib.pyplot as plt
from core.createSignal import *
from scipy.signal import lombscargle

"""
create a dictionary (for now its a fourier dictionary) for irregular spaced data and 
for an oversampling in frequency given by a factor n.
"""


class Dictionary(object):

    def __init__(self, times, nyquist_freq, n=5):
        self.t = times
        self.nyq = nyquist_freq
        self.n = n  # this could be from 5 to 10 as it say in lomb-scargle paper.
        self.df = self.get_df(times, n=n)

    def get_df(self, times, n=5):
        return 1/(n * (max(times) - min(times)))

    def min_freq(self):
        return 1/(max(self.t) - min(self.t))

    def get_frequencies(self, ac_component=False):
        """
        get the frequencies to use, the range it will be from 1/T to nyquist, but if ac_component its true
        then it will be from 0 to nyquist.
        :param ac_component: True if you want the AC component in frequencies
        """
        if ac_component:
            return np.linspace(0, self.nyq, int(self.nyq/self.df))
        else:
            return np.linspace(0, self.nyq, int((self.nyq - self.min_freq()) / self.df))

    def atom(self, freq, time):
        return np.exp(2j * np.pi * freq * time)

    def matrix(self, ac_component=False):
        freqs = self.get_frequencies(ac_component=ac_component)
        return self.atom(freqs, self.t[:, np.newaxis])