from typing import Tuple, Annotated
from pydantic import PositiveInt
import torch 
import numpy as np

def delta(shape: Tuple, n0: PositiveInt=0) -> np.ndarray:
    """
    Dirac delta function of arbitrary dimension.

    Arguments:
    shape (Tuple): shape of resultant signal
    n0 (PositiveInt=0): optional index of pulse (1.0)

    Returns:
    result (Iterable): signal of zeros with 1.0 at n0 in final dimension

    Example:
    >>> y = delta(shape=(8), n0=0)

    """
    result = np.zeros(shape=shape)
    result[Ellipsis, n0] = 1.0
    return result

def fir2spectrum(fir: np.ndarray, sample_rate: int=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert time domain Finite Impulse Response (FIR) to a 
    complex spectrum, with frequency vector.

    Arguments:
    fir (Iterable): finite impulse response
    sample_rate (int=1): sampling frequency (Hz)

    Returns:
    H (Iterable): complex half-spectrum corresponding to positive
                  frequencies
    f (Iterable): center frequency for each bin in `H`
    """
    H = np.fft.fft(fir)
    H = H[:, :H.shape[1]//2]
    f = np.linspace(0, sample_rate//2, H.shape[1], False)
    return H, f

def _fix_dims(signal: np.ndarray) -> np.ndarray:
    """
    Make sure a signal array is [C, T].
    """
    if signal.ndim() < 2:
        signal = np.expand_dims(signal, axis=0)
    elif signal.ndim() > 3:
        RuntimeError("`signal` must be 1 or 2 dimensional.")
    return signal

def is_stereo(signal: np.ndarray) -> bool:
    """
    Check to see if signal has two channels
    """
    signal = _fix_dims(signal)
    num_channels = signal.shape[0]
    
    return num_channels == 2
