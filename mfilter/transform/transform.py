import numpy as np


class FourierTransform(object):
    def __init__(self):
        self._frequency = []

    def forward(self, data, uniform=False, **kwargs):
        pass

    def backward(self, data, **kwargs):
        pass

    def get_frequency(self, **kwargs):
        pass

    def get_times(self, **kwargs):
        pass


class NFFT(FourierTransform):
    pass


class Regression(FourierTransform):

    def __init__(self, reg, freq):
        super().__init__()
        self.reg = reg
        self.freq = freq
        self.active = False

    def _set(self, data):
        self.reg.reset()
        self.reg.create_dict(data.times, self.freq)

    def forward(self, data, uniform=False, **kwargs):
        self.reg.set_coef(data.value)
        return self.reg.predict()

    def backward(self, data, **kwargs):
        self._set(data)
        return self.reg.get_ft(data)

    def get_frequency(self, **kwargs):
        return self.reg.frequency

    def get_times(self, **kwargs):
        return self.reg.time


class FFT(FourierTransform):
    def __init__(self, times, beta=2):
        super().__init__()
        self.beta = beta
        self.times = times
        self.reg_times = np.linspace(self.times.min(), self.times.max(), self.beta * len(self.times))

    def forward(self, data, uniform=False, **kwargs):
        tmp = np.fft.ifft(data.value * kwargs.get('N', None))
        if uniform:
            return tmp
        else:
            return np.interp(self.times, self.reg_times, tmp)

    def backward(self, data, **kwargs):
        return np.fft.fft(np.interp(self.reg_times, self.times.value, data.value)) / kwargs.get('N', None)

    def get_frequency(self, **kwargs):
        dt = (self.reg_times.max() - self.reg_times.min()) / len(self.reg_times)
        return np.fft.fftfreq(self.beta * kwargs.get('N', None), d=dt)

    def get_times(self, **kwargs):
        return self.times
