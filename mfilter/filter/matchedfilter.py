
from mfilter.types import TimeSeries, FrequencySeries, FrequencySamples
import numpy as np


def make_frequency_series(vec, frequency_grid=None, method="regression",
                          **kwargs):

    if isinstance(vec, FrequencySeries):
        return vec
    if isinstance(vec, TimeSeries):
        if not isinstance(frequency_grid, FrequencySamples):
            raise ValueError("to do the transfrom you need to ge a valid"
                             "FrequencySamples object")
        return vec.to_frequencyseries(frequency_grid=frequency_grid,
                                      method=method, **kwargs)
    else:
        raise TypeError("Can only convert a TimeSeries to  a FrequencySeries")


def sigmasq(htilde, psd=None, frequency_grid=None, method="regression",
            **kwargs):
    htilde = make_frequency_series(htilde, frequency_grid=frequency_grid,
                                   method=method, **kwargs)
    norm = 1

    if psd is None:
        sq = htilde.inner()
    else:
        assert psd.delta_f == htilde.delta_f
        assert (psd.frequencies == htilde.frequencies).all()
        sq = htilde.weighted_inner(weight=psd)

    return sq.real * norm


def sigma(htilde, psd=None, frequency_grid=None, method="regression"):
    return np.sqrt(sigmasq(htilde, psd=psd, frequency_grid=frequency_grid,
                           method=method))


def correlate(htilde: FrequencySeries, stilde: FrequencySeries) -> FrequencySeries:
    return FrequencySeries(stilde * htilde.conj(),
                           frequency_grid=htilde.frequency_object,
                           epoch=htilde.epoch)


def linear_filter(filter, data, frequency_grid=None,
                  method="regression", **kwargs):
    htilde = make_frequency_series(filter, frequency_grid=frequency_grid,
                                   method=method, **kwargs)
    stilde = make_frequency_series(data, frequency_grid=frequency_grid,
                                   method=method, **kwargs)

    qtilde = correlate(htilde, stilde)

    q = 2 * qtilde.to_timeseries(method=method, **kwargs) / data.average_fs

    if isinstance(data, TimeSeries):
        times = data.times
    else:
        times = kwargs.get("times", None)

    if times is None:
        raise ValueError("you need to give a valid times array or data as "
                         "TimeSeries object")

    if frequency_grid is None:
        frequency_grid = htilde.frequency_object

    return (TimeSeries(q, times=times),
            FrequencySeries(qtilde, frequency_grid=frequency_grid,
                            epoch=times.min()))


def matched_filter_core(template, data, psd=None, h_norm=None,
                        frequency_grid=None, method="regression", **kwargs):

    htilde = make_frequency_series(template, frequency_grid=frequency_grid,
                                   method=method, **kwargs)
    stilde = make_frequency_series(data, frequency_grid=frequency_grid,
                                   method=method, **kwargs)

    if len(htilde) != len(stilde):
        raise ValueError("length of template and data must match")

    if (htilde.frequencies != stilde.frequencies).all():
        raise ValueError("frequencies of sampling of template and data must "
                         "match")

    qtilde = correlate(htilde, stilde)

    if psd is not None:
        if isinstance(psd, FrequencySeries):
            if psd.delta_f == stilde.delta_f and psd.min_freq == stilde.min_freq:
                qtilde /= psd
            else:
                raise TypeError("PSD frequencies does not match data")
        else:
            raise TypeError("PSD must be a FrequencySeries")

    q = qtilde.to_timeseries(method=method, **kwargs)

    if h_norm is None:
        h_norm = sigmasq(htilde, psd=psd, frequency_grid=frequency_grid,
                         method=method, **kwargs)

    norm = 1 / np.sqrt(h_norm)

    if isinstance(data, TimeSeries):
        times = data.times
    else:
        times = kwargs.get("times", None)

    if times is None:
        raise ValueError("you need to give a valid times array or data as "
                         "TimeSeries object")

    if frequency_grid is None:
        frequency_grid = htilde.frequency_object

    return (TimeSeries(q, times=times),
            FrequencySeries(qtilde, frequency_grid=frequency_grid,
                            epoch=times.min()), norm)


def matched_filter(template, data, psd=None, frequency_grid=None,
                   method="regression", h_norm=None, unitary_energy=False,
                   **kwargs):
    stilde = make_frequency_series(data, frequency_grid=frequency_grid,
                                   method=method, **kwargs)

    snr, _, norm = matched_filter_core(template, stilde, psd=psd,
                                       frequency_grid=frequency_grid,
                                       method=method, h_norm=h_norm,
                                       **kwargs)
    if not unitary_energy:
        data_energy = sigmasq(stilde, psd=psd)
    else:
        data_energy = 1

    return snr * norm / np.sqrt(data_energy)


def match(vec1, vec2, psd=None, v1_norm=None, v2_norm=None,
          frequency_grid=None, method="regressor", **kwargs):

    stilde = make_frequency_series(vec2, frequency_grid=frequency_grid,
                                   method=method, **kwargs)

    snr, _, snr_norm = matched_filter_core(vec1, stilde, psd=psd,
                                           h_norm=v1_norm, **kwargs)

    max_snr, max_idx = snr.abs_max_loc()

    if v2_norm is None:
        v2_norm = sigmasq(stilde, psd=psd)

    return max_snr * snr_norm / np.sqrt(v2_norm), max_idx