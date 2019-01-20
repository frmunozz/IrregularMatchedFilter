import numpy as np

from imf.types import TimeSeries, FrequencySeries, Array, TimeSamples
from imf.regressions.regressors import BasicRegression
from imf.transform.transform import FourierTransformer


def matched_filter_core(times: TimeSamples, data: FrequencySeries,
                        template: FrequencySeries, tr: FourierTransformer, psd=None):
    """
    Core function to perfomr Matched Filter, it need as inputs the data and template, both as FrequencySeries,
    the Transformer to use and the PSD which can be an Array, Array-like, FrequencySeries or None type.

    Inside this we compute the Matched Filter operation and normalize using the corresponding units.

    This functions doesnt check the types of the parameters, the responsibility of give the correct types
    rely on the user. If you give wrong times, it can raise an error or just compute random stuff.

    :param times:               times to use,
                                    it must be a TimeSamples
    :param data:                data to match,
                                    it must be a FrequencySeries
    :param template:            template to match,
                                    it must be a FrequencySeries
    :param tr:                  Transformer to use,
                                    it must be a FourierTransformer
    :param psd:                 Power Spectral Density to use,
                                    it can be an Array, Array-like
                                    (list or numpy array), FrequencySeries or None
    """

    fs = times.average_fs
    df = data.df

    stilde = data / fs
    htilde = template / fs
    if psd is None:
        psd = 1

    optimal = stilde * htilde.conjugate() / psd
    optimal_time = 2 * fs * FrequencySeries(optimal, frequency_grid=data.frequencies, epoch=data.epoch).to_timeseries(tr)

    sigmasq = (htilde * htilde.conjugate() / psd).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))

    return TimeSeries(optimal_time / sigma, times=times)
