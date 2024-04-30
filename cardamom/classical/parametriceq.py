import os
from typing import Tuple, Iterable
import scipy.signal
import torch
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import librosa

from utils import delta, fir2spectrum

"""
Parametric Equalizer 

A "series connection of first-and second-order shelving and peak
filters, which are controlled independently" [1, p. 63]

[1] Zölzer, Udo, ed. DAFX Digital Audio Effects. 2nd ed. Chichester, 
    West Sussex, UK: John Wiley & Sons, 2011.
"""

def _parametric_constants(fs, fc, gain) -> Tuple[float, float]:
    """
    Constants for 2nd order peak and shelf filters.
    See Table 2.3 in [1], p. 23.

    Returns:
    V0 (float)
    K (float)
    """
    V0 = 10.0**(gain/20)
    K = np.tan(np.pi * fc / fs)
    return K, V0

def lowshelf(fs: int, fc: float, gain: float) -> Tuple[Iterable, Iterable]:
    """
    Coefficients for a biquad lowshelf filter.

    Arguments:
    fs (int): sampling rate (Hz)
    fc (float): cutoff frequency (Hz)
    gain (float): gain (dB)

    Returns: 
    b (list): feedforward coefficients
    a (list): feedback coefficients

    Example: 
    >>> import scipy
    >>> import matplotlib.pyplot as plt
    >>> b, a = lowshelf(32000, 200, -6)
    >>> w, h = scipy.signal.freqz(b, a)
    >>> plt.plot(w, librosa.amplitude_to_db(h))

    """

    K, V0 = _parametric_constants(fs, fc, gain)

    # low shelf
    denom = (1 + np.sqrt(2)*K + K**2)
    b0 = (1 + np.sqrt(2 * V0)*K + V0*K**2) / denom
    b1 = 2*(V0 * K**2 - 1) / denom
    b2 = (1 - np.sqrt(2*V0)*K + V0*K**2) / denom
    a1 = 2*(K**2 - 1) / denom
    a2 = (1 - np.sqrt(2)*K + K**2) / denom

    return [b0, b1, b2], [1, a1, a2]

def highshelf(fs: int, fc: float, gain: float) -> Tuple[Iterable, Iterable]:
    """
    Coefficients for a biquad highshelf filter.

    Arguments:
    fs (int): sampling rate (Hz)
    fc (float): cutoff frequency (Hz)
    gain (float): gain (dB)

    Returns: 
    b (list): feedforward coefficients
    a (list): feedback coefficients

    Example: 
    >>> import scipy
    >>> import matplotlib.pyplot as plt
    >>> b, a = highshelf(32000, 200, -6)
    >>> w, h = scipy.signal.freqz(b, a)
    >>> plt.plot(w, librosa.amplitude_to_db(h))
    """

    K, V0 = _parametric_constants(fs, fc, gain)

    # high shelf
    denom = (1 + np.sqrt(2)*K + K**2)
    b0 = (V0 + np.sqrt(2*V0)*K + K**2) / denom
    b1 = 2*(K**2 - V0) / denom
    b2 = (V0 - np.sqrt(2*V0)*K + K**2) / denom
    a1 = 2*(K**2 - 1) / denom
    a2 = (1 - np.sqrt(2)*K + K**2) / denom

    return [b0, b1, b2], [1, a1, a2]

def peak(fs: int, fc: float, gain: float, q: float) -> Tuple[Iterable, Iterable]:
    """
    Coefficients for a peak filter.

    Arguments:
    fs (int): sampling rate (Hz)
    fc (float): cutoff frequency (Hz)
    gain (float): gain (dB)
    q (float): resonance/bandwidth

    Returns: 
    b (list): feedforward coefficients
    a (list): feedback coefficients

    Example: 
    >>> import scipy
    >>> import matplotlib.pyplot as plt
    >>> b, a = peak(32000, 200, -6)
    >>> w, h = scipy.signal.freqz(b, a)
    >>> plt.plot(w, librosa.amplitude_to_db(h))
    """

    K, V0 = _parametric_constants(fs, fc, gain)

    denom = (1 + (1/q)*K + K**2)
    b0 = (1 + (V0/q)*K + K**2) / denom
    b1 = 2*(K**2 - 1) / denom
    b2 = (1 - (V0/q)*K + K**2) / denom
    a1 = 2*(K**2 - 1) / denom
    a2 = (1 - (1/q)*K + K**2) / denom

    return [b0, b1, b2], [1, a1, a2]

class Filter():
    """
    Wrapper for `scipy.signal.lfilter` with multichannel
    processing.
    """
    def __init__(self, b, a):
        self.b = b
        self.a = a

    def process(self, x: np.ndarray) -> np.ndarray:
        """
        Arguments:
        x (np.ndarray): [C, T] signal

        Returns :
        y (np.ndarray): [C, T] filtered signal
        """
        num_channels = x.shape[0]
        num_samples = x.shape[1]

        y = np.zeros_like(x)
        for channel in range(num_channels):
            y[channel, :] = scipy.signal.lfilter(self.b, self.a, x[channel, :])

        return y

class ParametricEqualizer():
    """
    Parametric equalizer consister of the following LTI systems in 
    series:
        Low Shelf -> Peak -> Peak ... -> Peak -> High Shelf
    """

    def __init__(self, sample_rate: int, num_filters: int):
        self.sample_rate = sample_rate
        self.num_filters = num_filters
        self._init_filters()
        
    def _init_filters(self):
        self.center_frequencies = np.geomspace(start=0.1 * self.sample_rate//2, 
                                               stop=0.9 * self.sample_rate//2,
                                               num=self.num_filters)
        self.filters = list()
        for n, fc in enumerate(self.center_frequencies):
            if n == 0:
                b, a = lowshelf(self.sample_rate, fc, 0.0)
            elif n == len(self.center_frequencies)-1:
                b, a = highshelf(self.sample_rate, fc, 0.0)
            else:
                b, a = peak(self.sample_rate, fc, 0.0, np.sqrt(2)/2)
            self.filters.append(Filter(b, a))

    

    def process(self, x: np.ndarray) -> np.ndarray:
        """
        Apply parametric filters to input audio by cascading 
        second order sections.

        Arguments:
        x (np.ndarray): [C, T] signal

        Returns :
        y (np.ndarray): [C, T] filtered signal
        """
        num_channels = x.shape[0]
        num_samples = x.shape[1]

        y = np.zeros_like(x)
        for n, filt in enumerate(self.filters):
            if n == 0:
                y = filt.process(x)
            else:
                y = filt.process(y)
        
        return y

num_samples = 1024
fs = 8000
eq = ParametricEqualizer(sample_rate=fs, num_filters=3)

d = delta(shape=(1, num_samples))
h = eq.process(d)

fig, axs = plt.subplots(2, 1, sharex=True)
H, f = fir2spectrum(h, sample_rate=fs)
axs[0].plot(f, np.abs(H[0, :]))

for filt in eq.filters:
    h = filt.process(d)
    H, f = fir2spectrum(h, sample_rate=fs)

    axs[1].plot(f, np.abs(H[0, :]))
plt.show()

# fs = 8000
# fc = fs/4
# gain = -10 # gain in dB

# b, a = peak(fs, fc, gain, 10)
# w, h = scipy.signal.freqz(b, a)
# plt.plot(w, librosa.amplitude_to_db(h))

plt.show()

