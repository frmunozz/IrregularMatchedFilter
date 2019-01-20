import numpy as np


class FourierTransformer(object):
    def __init__(self):
        """
        Transformer object to perform Fourier Transform

        """
        self._frequency = []

    def forward(self, data, **kwargs):
        """
        compute Forward Fourier Transform (from frequency domain
        to time domain)

        :param data:        data to transform
        :param kwargs:      additional parameters that may be used in
                            different methods.
        """
        pass

    def backward(self, data, **kwargs):
        """
        compute Backward Fourier Transform (from time domain to
        frequency domain)

        :param data:        data to transform
        :param kwargs:      additional parameters that may be used in
                            different methods.
        """
        pass

    def get_frequency(self, **kwargs):
        """
        method that return the frequency grid used to transform

        :param kwargs:      additional parameters that may be used in
                            different methods.
        """
        pass

    def get_times(self, **kwargs):
        """
        method that return the times grid used to transform

        :param kwargs:      additional parameters that may be used in
                            different methods.
        """
        pass


class RegressionTransformer(FourierTransformer):

    def __init__(self, reg, freq):
        """
        Transformer sub-class from Regression method,
            perform Fourier Transform using Linear Regression to
            compute fourier coefficients.

        :param reg:         The Regression Object.
        :param freq:        The FrequencySamples Object.
        """
        super().__init__()
        self.reg = reg
        self.freq = freq
        self.active = False

    def _set(self, data):
        """
        inner method used to reset a Regression Object
        and create a new Dictionary to use with.

        :param data:        data to use in the Regression Object,
                            which will use the TimeSamples stored in
                            the data TimeSeries.
        """
        self.reg.reset()
        self.reg.create_dict(data.times, self.freq)

    def forward(self, data, **kwargs):
        self.reg.set_coef(data)
        return self.reg.predict(new_coef=kwargs.get('new_coef', True))

    def backward(self, data, **kwargs):
        self._set(data)
        return self.reg.get_ft(data)

    def get_frequency(self, **kwargs):
        return self.freq

    def get_times(self, **kwargs):
        return self.reg.time


class FFTTransformer(FourierTransformer):
    def __init__(self, times):
        super().__init__()
        self.times = times

    def forward(self, data, new_coef=True, **kwargs):
        return np.fft.ifft(data.data)

    def backward(self, data, **kwargs):
        return np.fft.fft(data.data)

    def get_frequency(self, **kwargs):
        from ..types.frequencyseries import FrequencySamples

        freqs = np.fft.fftfreq(kwargs.get('N', None))
        df = np.abs(freqs[1] - freqs[0])
        return FrequencySamples(freqs*self.times.average_fs, df=df)

    def get_times(self, **kwargs):
        return self.times
