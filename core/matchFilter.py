import numpy as np
from scipy import signal
from core.getPSD import get_psd
import pdb


class MatchFilter(object):

    def __init__(self, data, template, dt, alpha=0.5):
        self.dt = dt
        self.df = 2 * np.pi / len(data)
        self.fs = int(1 / dt)
        self.data = data
        self.template = template
        self.freqs = self.compute_freqs()
        self.dwindow = signal.tukey(template.size, alpha=alpha)

    def compute_freqs(self):
        return np.fft.fftfreq(self.template.size) * self.fs

    def product(self, data_fft, template_fft, power_vec):
        return data_fft * template_fft.conjugate() / power_vec

    def make_fft(self, noise):
        template_fft = np.fft.fft(self.template * self.dwindow) / self.fs
        data_fft = np.fft.fft(self.data * self.dwindow) / self.fs
        power_vec, _ = get_psd(noise, int(1 / self.dt), freqs_to_interp=self.freqs)
        return template_fft, data_fft, power_vec

    def SNR(self, noise, normalize=True, shift=True): # as defined from LIGO
        template_fft, data_fft, power_vec = self.make_fft(noise)
        optimal = 2 * self.fs * np.fft.ifft(self.product(data_fft, template_fft, power_vec))
        # in order to get an SNR of 1 when the template is at a time of where the signal is only noise,
        # we divide by this sigma:
        if normalize:
            sigmasq = 1 * (template_fft * template_fft.conjugate() / power_vec).real.sum() * self.df
            sigma = np.abs(sigmasq)
        else:
            sigma = 1

        SNR = optimal / sigma
        # this wil give us an array of matches between the signal and the template representing
        # the offset in time, this mean that we will get the maximun SNR at offset 0 and the same value at the end
        # of the array (maximun offset), this is because is cyclical, so in order to get the maximun SNR at an
        # offset of time between the minimum (offset 0) and maximum, we move the template peak to the end and
        # since we usually have the peak of the template at the middle of time, we do:
        if shift:
            peaksample = int(self.template.size / 2)  # location of peak in the template
            SNR= np.roll(SNR, peaksample)  # we shift the SNR because we want to shift0
                                                            # the time of the template.

        return np.abs(SNR)
