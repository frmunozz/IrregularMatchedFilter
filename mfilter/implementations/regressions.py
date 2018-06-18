from sklearn import linear_model
from mfilter.types.arrays import Array
import numpy as np


def _cast_into_ft(coefs):
    n_freqs = int(len(coefs) / 2)
    ft = 1j * np.zeros(n_freqs)
    for i in range(n_freqs):
        ft[i] = coefs[i] + 1j * coefs[i + n_freqs]
    return ft


def _split_fourier_dict(F):
    return np.hstack((F.real, F.imag))


class Dictionary(object):
    def __init__(self, times, frequencies):
        if not isinstance(times, Array):
            times = Array(np.array(times))

        if not isinstance(frequencies, Array):
            frequencies = Array(np.array(frequencies),
                                delta=frequencies[1] - frequencies[0])
        self._t = times
        self._f = frequencies
        self._dict = self.compute_dict()

    def compute_dict(self, times=None, frequency=None):
        if isinstance(times, Array):
            self._t = times

        if isinstance(frequency, Array):
            self._t = frequency

        matrix = self._t.data.reshape(-1, 1) * self._f.data
        return np.exp(2j * np.pi * matrix)

    @property
    def frequency(self):
        return self._f.data

    @property
    def time(self):
        return self._t.data

    @property
    def matrix(self):
        return self._dict

    @property
    def splited_matrix(self):
        return _split_fourier_dict(self._dict)

    @property
    def df(self):
        return self._f.delta


class BasicRegression(object):
    def __init__(self, overfit=True, phi: Dictionary = None):
        self._ovfit = overfit
        self._reg = self._get_instance()

        self._dict = Dictionary([-1, -1], [-1, -1])
        self._valid = False

        if phi is not None:
            self.set_dict(phi)

    def _validate_phi(self, phi_dict):
        if phi_dict is None:
            return

        if isinstance(phi_dict, np.ndarray):
            if len(phi_dict) == 2:
                phi_dict = Dictionary(phi_dict[0], phi_dict[1])

        elif not isinstance(phi_dict, Dictionary):
            raise ValueError("'phi_dict' must be either a numpy array"
                             "or an instance of Dictionary")
        self._dict = phi_dict
        self._valid = True

    def _get_instance(self, cv=3):
        pass

    @property
    def coef(self):
        return self._reg.coef_

    @property
    def dict(self):
        return self._dict

    @property
    def df(self):
        return self.dict.df

    @property
    def frequency(self):
        return self.dict.frequency

    @property
    def time(self):
        return self._dict.time

    def set_dict(self, times, frequency):
        phi = Dictionary(times, frequency)
        self._validate_phi(phi)

    def fit(self, y, phi: Dictionary = None):
        self._validate_phi(phi)
        if not self._valid:
            raise ValueError("regressor doesn't have a valid dictionary")

        self._reg.fit(self._dict.splited_matrix, y)

    def get_ft(self, y: Array, phi: Dictionary = None):
        self.fit(y.data, phi=phi)
        return _cast_into_ft(self.coef)

    def predict(self, phi: Dictionary =None):
        return self._reg.predict(phi)

    def reconstruct(self, beta: Array):
        return np.dot(self._dict.matrix, beta.data)


class RidgeRegression(BasicRegression):
    def __init__(self, alpha=1, overfit=True, solver='auto',
                 phi: Dictionary = None):
        self._a = alpha
        self._solver = solver
        super().__init__(overfit=overfit, phi=phi)

    def _get_instance(self, cv=3):
        if self._ovfit:
            return linear_model.Ridge(alpha=self._a, solver=self._solver)
        else:
            return linear_model.RidgeCV(alphas=self._a, cv=cv)


class LassoRegression(BasicRegression):
    def __init__(self, alpha=1, overfit=True, phi: Dictionary = None):
        self._a = alpha
        super().__init__(overfit=overfit, phi=phi)

    def _get_instance(self, cv=3):
        if self._ovfit:
            return linear_model.Lasso(alpha=self._a)
        else:
            return linear_model.LassoCV(alphas=self._a, cv=cv)


class ElasticNetRegression(BasicRegression):
    def __init__(self, alpha=1, l1_ratio=0.5, overfit=True,
                 phi: Dictionary = None):
        self._a = alpha
        self._l1 = l1_ratio
        super().__init__(overfit=overfit, phi=phi)

    def _get_instance(self, cv=3):
        if self._ovfit:
            return linear_model.ElasticNet(alpha=self._a, l1_ratio=self._l1)
        else:
            return linear_model.ElasticNetCV(l1_ratio=self._l1, alphas=self._a)

