from sklearn import linear_model
from sklearn.preprocessing import scale
from imf.types import Array
from imf.regressions.dictionaries import Dictionary
import numpy as np


def _cast_into_ft(coefs):
    """
    transform real coefficient of the Fourier Series to Complex coefficients
    of the Fourier Transform.

    :param coefs:       the real coefficients.
    :return:            the complex coefficients.
    """
    n_freqs = int(len(coefs) / 2)
    ft = 1j * np.zeros(n_freqs)
    for i in range(n_freqs):
        ft[i] = coefs[i] - 1j * coefs[i + n_freqs]
    return ft


def split_ft(ft):
    """
    transform complex coefficients of the Fourier Transform to real coefficients
    of the Fourier Series.

    :param ft:          the complex coefficients.
    :return:            the real coefficients.
    """
    coefs = np.zeros(len(ft)*2)
    for i in range(len(ft)):
        coefs[i] = ft[i].real
        coefs[i + len(ft)] = -ft[i].imag
    return coefs


class BasicRegression(object):
    def __init__(self, overfit=True, phi: Dictionary = None):
        """
        Basic Regression class,
            for the creation of a Linear Regression instance,
            here we define the common methods of all regressions types.

        :param overfit:     True if our Regression need to be over fitted.
        :param phi:         Dictionary object (matrix) to use in the Regression
        """
        self._ovfit = overfit
        self._reg = self._get_instance()
        self._dict = Dictionary([-1, -1], [-1, -1])
        self._valid = False
        self._validate_phi(phi)
        self.coef_ = None
        self.valid = False

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
        """
        Method used on the child-classes to generate the
        specific Regression method.

        :param cv:
        """
        pass

    @property
    def coef(self):
        """
        :return:            coeficients of the Regression, if there is no coefficients
                            (no .fit done so far), the return an array of zeros.
        """
        if self.coef_ is None:
            return np.zeros(len(self._dict.frequency))
        else:
            return self.coef_

    @property
    def dict(self):
        """
        :return:            Dictionary object to use in the Regression
        """
        return self._dict

    @property
    def df(self):
        """
        :return:            frequency step from the frequency used on the Dictionary.
        """
        return self.dict.df

    @property
    def frequency(self):
        """
        :return:            Frequencies used on the Dictionary
        """
        return self.dict.frequency

    @property
    def time(self):
        """
        :return:            Times used in the Dictionary
        """
        return self._dict.time

    def create_dict(self, times, frequency):
        """
        Create a new Dictionary to use on the Regression method.

        :param times:       the new times to use (TimeSamples)
        :param frequency:   the new frequencies to use (FrequencySamples)
        """
        phi = Dictionary(times, frequency)
        self._validate_phi(phi)
        self.valid = True

    def set_dict(self, phi: Dictionary):
        """
        Set a new dictionary as the default dictionary to use by the Regression.

        :param phi:         the new Dictionary
        """
        self._validate_phi(phi)

    def scale(self):
        if not self._valid:
            raise ValueError("regressor doesn't have a valid dictionary")
        scale(self._dict.splited_matrix)

    def fit(self, y: np.ndarray):
        """
        Fit coefficients to input data using the default dictionary.

        :param y:           input data
        """
        if not self._valid:
            raise ValueError("regressor doesn't have a valid dictionary")

        n = self._dict.shape(split_matrix=True)[1]
        if self.coef_ is None or len(self.coef_) != n:
            self.reset()

        self._reg.fit(self._dict.splited_matrix, y, coef_init=self.coef_)
        self.coef_ = self._reg.coef_

    def get_ft(self, y: Array):
        """
        get the Fourier Coefficients for a given input TimeSeries using
        the default Dictionary

        :param y:           input TimeSeries
        :return:            Fourier Coefficients (FrequencySeries)
        """
        self.fit(y.data)
        return _cast_into_ft(self.coef)

    def predict(self, phi: Dictionary =None, new_coef=True):
        """
        Predict values from the current computed coefficients,
        This is transform from frequency domain to time domain.

        :param phi:         the Dictionary to use
        :param new_coef:    True if we want to use new computed coefficients.
        :return:            TimeSeries with the predicted data.
        """
        if phi is None:
            phi = self.dict
        if new_coef:
            self._reg.coef_ = self.coef_
        return self._reg.predict(phi.splited_matrix)

    def reconstruct(self, frequency_data: Array):
        return np.dot(self._dict.matrix, frequency_data.data)

    def score(self, y: Array):
        """
        :param y:           input TimeSeries
        :return:            R2 score from the predicted TimeSeries compared
                            to the input
        """
        return self._reg.score(self._dict.splited_matrix, y.data)

    def reset(self):
        """
        reset the coefficients of the regressions and set them to 0.
        """
        self.coef_ = np.zeros(self.dict.shape(split_matrix=True)[1])

    def set_coef(self, ft):
        """
        set the coefficients of the regressions from a given value
        :param ft: value (complex fourier coefficients) to use.
        """
        self.coef_ = split_ft(ft)


class RidgeRegression(BasicRegression):
    def __init__(self, alpha=1, overfit=True, solver='auto',
                 phi: Dictionary = None):
        """
        Ridge Regression

        :param alpha:
        :param overfit:
        :param solver:
        :param phi:
        """
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
                 penalty="l2", l1_ratio=0.5, overfit=True, phi: Dictionary = None):
        """
        Stochastic Gradient Descent Class,
            sub-class of Basic Regression, it will perform the Linear Regression using
            Stochastic Gradient Descent.

        :param alpha:
        :param max_iter:
        :param tol:
        :param penalty:
        :param l1_ratio:
        :param overfit:
        :param phi:
        """
        self._a = alpha
        self._max_iter = max_iter
        self._tol = tol
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        super().__init__(overfit, phi)

    def _get_instance(self, cv=3):
        if self._ovfit:
            return linear_model.SGDRegressor(alpha=self._a,
                                             max_iter=self._max_iter,
                                             tol=self._tol,
                                             shuffle=True,
                                             penalty=self.penalty,
                                             l1_ratio=self.l1_ratio,
                                             average=False,
                                             learning_rate='invscaling',
                                             eta0=0.01,
                                             power_t=0.25)
        else:
            raise ValueError("method dosent have a cross validation implementation")


