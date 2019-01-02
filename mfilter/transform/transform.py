import numpy as np


class FourierTransform(object):
    def __init__(self):
        self._frequency = []

    def forward(self, data, uniform=False, new_coef=True, **kwargs):
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

    def forward(self, data, uniform=False, new_coef=True, **kwargs):
        self.reg.set_coef(data)
        return self.reg.predict(new_coef=new_coef)

    def backward(self, data, **kwargs):
        self._set(data)
        return self.reg.get_ft(data)

    def get_frequency(self, **kwargs):
        return self.freq

    def get_times(self, **kwargs):
        return self.reg.time


class FFT(FourierTransform):
    def __init__(self, times, beta=2):
        super().__init__()
        # self.beta = beta
        self.times = times
        # self.reg_times = np.linspace(self.times.min(), self.times.max(), self.beta * len(self.times))

    def forward(self, data, uniform=False, new_coef=True, **kwargs):
        return np.fft.ifft(data.value)
        # if uniform:
        #     return tmp
        # else:
            # return np.interp(self.times, self.reg_times, tmp)

    def backward(self, data, **kwargs):
        # return np.fft.fft(np.interp(self.reg_times, self.times.value, data.value)) / kwargs.get('N', 1)
        return np.fft.fft(data.value)

    def get_frequency(self, **kwargs):
        # dt = (self.reg_times.max() - self.reg_times.min()) / len(self.reg_times)
        # dt=1
        # return np.fft.fftfreq(self.beta * kwargs.get('N', len(self.times)), d=dt)
        from ..types.frequencyseries import FrequencySamples
        freqs = np.fft.fftfreq(kwargs.get('N', None))
        df = np.abs(freqs[1] - freqs[0])
        return FrequencySamples(freqs*self.times.average_fs, df=df)

    def get_times(self, **kwargs):
        return self.times
    #
    # def get_fs(self, **kwargs):
    #     return 1 / np.abs(self.reg_times[1] - self.reg_times[0])
