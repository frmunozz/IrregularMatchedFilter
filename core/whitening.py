import numpy as np
from core.getPSD import get_psd


class Whitening(object):
    """
    Class to whiten signals, i.e., reduce the noise fluctuations.
    """

    def __init__(self, data, fs):
        self.data = data
        self.fs = fs
        self.dt = 1 / self.fs
        self.freqs = np.fft.rfftfreq(len(self.data), self.dt)

    def whiten(self, noise=None):
        """
        do the whitening fo the signal, here it can receive a noise or can estimate it.
        :param noise:
        :return:
        """
        fft_data = np.fft.rfft(self.data)
        norm = 1. / np.sqrt(1. / (self.dt * 2))  # normalized by the nyquist frequency(?)
        if noise is not None:
            psd = get_psd(noise, self.fs, freqs_to_interp=self.freqs)
        else:
            # usually the noise is much higher than the signal so the psd of the entire
            # signal is approx. the psd of the noise.
            psd = get_psd(self.data, self.fs, freqs_to_interp=self.freqs)
        white_fft_data = fft_data / np.sqrt(np.abs(psd)) * norm
        white_data = np.fft.irfft(white_fft_data, n=len(self.data))
        return white_data
