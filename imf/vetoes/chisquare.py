from imf.filter import matched_filter_core
from imf.types import FrequencySeries, TimeSeries
import numpy as np


def power_chisq(htilde, stilde, num_bins, times, psd=None,
                method="regression", **kwargs):

    bins = power_chisq_bins(htilde, num_bins, psd=psd, method=method, **kwargs)

    snr, corr, norm = matched_filter_core(htilde, stilde, psd=psd,
                                          times=times, method=method, **kwargs)

    return power_chisq_from_precomputed(corr, snr, norm, bins, times,
                                        method=method, **kwargs), len(bins)


def power_chisq_bins(htilde, num_bins, psd=None, method="regression", **kwargs):
    sigma_vec = sigmasq_series(htilde, psd=psd)

    return power_chisq_bins_from_sigmasq_series(sigma_vec, num_bins)


def sigmasq_series(htilde, psd=None):
    autocorr = htilde.conj() * htilde
    if psd is not None:
        autocorr /= psd
    return autocorr.cumsum()


def power_chisq_bins_from_sigmasq_series(sigma_vec, num_bins):
    sigmasq = sigma_vec[len(sigma_vec ) -2]
    edge_vec = np.arange(0, num_bins) * sigmasq / num_bins
    bins = np.searchsorted(sigma_vec, edge_vec, side='right')
    bins = np.append(bins, len(sigma_vec) - 1)
    bins = np.unique(bins)
    #     if len(bins) != num_bins + 1:
    #         print("using {} bins instead of {}".format(len(bins), num_bins))
    return bins


def power_chisq_from_precomputed(corr, snr, norm, bins, times,
                                 method="regression", **kwargs):
    qtilde = FrequencySeries(np.zeros(len(corr)),
                             frequency_grid=corr.frequency_object,
                             dtype=corr.dtype,
                             epoch=corr.epoch)
    chisq = TimeSeries(np.zeros(len(snr)), times=snr.times,
                       dtype=snr.dtype, epoch=snr.epoch)
    num_bins = len(bins) - 1

    for j in range(num_bins):
        k_min = int(bins[j])
        k_max = int(bins[ j +1])
        qtilde[k_min:k_max] = corr[k_min:k_max]
        q = qtilde.to_timeseries(method=method, times=times, **kwargs)
        qtilde.fill(0)
        chisq += q.squared_norm()

    chisq = (chisq * num_bins - snr.squared_norm()) * (norm ** 2)
    chisq = TimeSeries(chisq, times=snr.times, epoch=snr.epoch)
    return chisq


def weighted_snr(snr, chisq):
    for i in range(len(chisq)):
        if chisq[i] > 1:
            snr[i] /= ((1 + chisq[i]**3) / 2.0)**(1.0 / 6)

    return snr