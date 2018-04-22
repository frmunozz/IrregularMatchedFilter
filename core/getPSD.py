import numpy as np
import matplotlib.mlab as mlab
import pdb


def get_psd(data, fs, freqs_to_interp=None):
    this_window = np.blackman(int(4 * fs))
    psd, freq = mlab.psd(data, Fs=fs, NFFT=int(4*fs), noverlap=int(2*fs))
    if freqs_to_interp is not None:
        return np.interp(freqs_to_interp, freq, psd), freq
    else:
        return np.sqrt(np.abs(psd)), freq
