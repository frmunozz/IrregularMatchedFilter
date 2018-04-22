# coding: utf-8

import numpy as np


class Signal(object):
    def __init__(self, frequencies=None, weights=None, noise=None):
        """
        it will generate a signal usign the frequencies and the weight from input.
        the frecuencies should be in Hertz and the weight is a constante of "amplitud" 
        of every sub-signal associated to avery frequency.
        weight should be a number between 0 and 1
        """
        if weights is None:
            weights = [1]
        if frequencies is None:
            frequencies = [1]
        self.frequencies = frequencies
        self.weights = weights
        self.noise = noise
        self.amp = 1
    
    def noise_samples(self):
        return self.noise
    
    def sin_samples(self, times, with_noise=False):
        y = np.zeros(times.shape[-1])
        for i in range(len(self.frequencies)):
            y += self.weights[i] * np.sin(2 * np.pi * self.frequencies[i] * times)
        if with_noise:
            y += self.noise
        return self.amp * y
    
    def square_samples(self, times, with_noise=False):
        y = np.zeros(len(times))
        for i in range(len(self.frequencies)):
            y += self.weights[i] * np.sign(np.sin(2 * np.pi * self.frequencies[i] * times))
        if with_noise:
            y += self.noise
        return self.amp * y
    
    def gaussian_samples(self):
        pass

