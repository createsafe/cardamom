import os
import warnings
from pydantic import Field
from typing import Tuple, Iterable
import scipy.signal
import torch
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import librosa

from cardamom.classical.utils import _fix_dims, is_stereo

class MidSide():
    """
    Mid-Side processing for stereo spread.

    Arguments:
    spread (float): amount of spread applied.
                    spread=0 for no spread
                    spread=-1 for mono
                    spread=1 for side
    """
    def __init__(self, spread: float = Field(0.0, ge=-1.0, le=1.0)):
        self.spread = spread

    def process(self, signal):
        signal = _fix_dims(signal)
        
        if is_stereo(signal):
            m = (np.sqrt(-self.spread + 1)) * np.sum(signal, axis=0)
            d = np.sqrt((self.spread+1)) * np.diff(signal, axis=0)
            l = m + d
            r = m - d
            return 0.5 * np.hstack((l, r))
        else:
            warnings.warn("MidSide not meaningful for mono signal")
            return signal