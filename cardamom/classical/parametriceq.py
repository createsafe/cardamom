import os
from typing import Tuple, Iterable
import scipy.signal
import torch
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import librosa

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

fs = 8000
fc = fs/4
gain = -10 # gain in dB

b, a = peak(fs, fc, gain, 10)
w, h = scipy.signal.freqz(b, a)
plt.plot(w, librosa.amplitude_to_db(h))

plt.show()

