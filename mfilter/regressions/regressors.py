from sklearn import linear_model
from mfilter.types import Array
from mfilter.regressions.dictionaries import Dictionary
import numpy as np


def _cast_into_ft(coefs):
    n_freqs = int(len(coefs) / 2)
    ft = 1j * np.zeros(n_freqs)
    for i in range(n_freqs):
        ft[i] = coefs[i] + 1j * coefs[i + n_freqs]
    return ft


def split_ft(ft):
    coefs = np.zeros(len(ft)*2)
    for i in range(len(ft)):
        coefs[i] = ft[i].real
        coefs[i + len(ft)] = ft[i].imag
    return coefs


class BasicRegression(object):
    def __init__(self, overfit=True, phi: Dictionary = None):
        self._ovfit = overfit
        self._reg = self._get_instance()
        self._dict = Dictionary([-1, -1], [-1, -1])
        self._valid = False
        self._validate_phi(phi)
        self.coef_ = None

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
        if self.coef_ is None:
            return np.zeros(len(self._dict.frequency))
        else:
            return self.coef_

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

    def fit(self, y: np.ndarray, phi: Dictionary = None):
        self._validate_phi(phi)
        if not self._valid:
            raise ValueError("regressor doesn't have a valid dictionary")

        n = self._dict.shape(splited=True)[1]
        if self.coef_ is None or len(self.coef_) != n:
            self.coef_ = np.zeros(self._dict.shape(splited=True)[1])

        self._reg.fit(self._dict.splited_matrix, y, coef_init=self.coef_)
        # self._reg.fit(self._dict.splited_matrix, to_fit)

        self.coef_ = self._reg.coef_

    def get_ft(self, y: Array, phi: Dictionary = None):
        self.fit(y.value, phi=phi)
        return _cast_into_ft(self.coef)

    def predict(self, phi: Dictionary =None):
        return self._reg.predict(phi.splited_matrix)

    def reconstruct(self, frequency_data: Array):
        return np.dot(self._dict.matrix, frequency_data.value)

    def score(self, y: Array):
        return self._reg.score(self._dict.splited_matrix, y.value)


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
                 phi: Dictionary = None, flags= None):
        self._a = alpha
        self._l1 = l1_ratio
        super().__init__(overfit=overfit, phi=phi)

    def _get_instance(self, cv=3):
        if self._ovfit:
            return linear_model.ElasticNet(alpha=self._a, l1_ratio=self._l1)
        else:
            return linear_model.ElasticNetCV(l1_ratio=self._l1, alphas=self._a)


class SGDRegression(BasicRegression):
    def __init__(self, alpha=0.0001, max_iter=500, tol=0.01,
                 overfit=True, phi: Dictionary = None):
        self._a = alpha
        self._max_iter = max_iter
        self._tol = tol
        super().__init__(overfit, phi)

    def _get_instance(self, cv=3):
        if self._ovfit:
            return linear_model.SGDRegressor(alpha=self._a,
                                             max_iter=self._max_iter,
                                             tol=self._tol,
                                             shuffle=True,
                                             penalty="elasticnet",l1_ratio=0.5)
        else:
            raise ValueError("method dosent have a cross validation implementation")


