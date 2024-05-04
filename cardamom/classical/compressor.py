import os
import warnings
from pydantic import Field
from typing import Tuple, Iterable
import torch
import torchaudio
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import librosa

from cardamom.classical.duration import Duration, DurationUnits
from cardamom.classical.utils import _fix_dims

class PeakFollower():
    """
    Auto regressive envelope follower.

    Example:
    >>> audiofile = "80bpm.wav"
    >>> signal, sample_rate = librosa.load(audiofile, sr=None, mono=False)
    
    >>> follower = PeakFollower(sample_rate=sample_rate,
                            attack_time=Duration(30.0, DurationUnits.MILLISECONDS),
                            release_time=Duration(300.0, DurationUnits.MILLISECONDS))
    >>> envelope = follower.process(signal) 
    >>> ax = plt.gca()
    >>> ax2 = ax.twinx()
    >>> ax.plot(signal[0, :])
    >>> ax2.plot(envelope[0, :], 'r')
    >>> plt.show()
    """
    def __init__(self, 
                 sample_rate: int, 
                 *, 
                 attack_time: Duration,
                 release_time: Duration):
        
        self.sample_rate = sample_rate
        self.attack = self._ms2coefficient(attack_time.get_ms())
        self.release = self._ms2coefficient(release_time.get_ms())
        self.last_envelope = 0.0

    def _ms2coefficient(self, ms):
        lti_time_constant = np.log(.368)
        return np.exp(lti_time_constant / (ms * self.sample_rate * 0.001))
    
    def process(self, signal):
        """
        Apply envelope follower to signal

        Arguments:
        x (np.ndarray): [C, T] signal

        Returns :
        y (np.ndarray): [C, T] envelope of signal
        """

        signal = _fix_dims(signal)
        num_channels = signal.shape[0]
        num_samples = signal.shape[1]

        self.last_envelope = np.zeros(shape=(num_channels))

        # rectify signal
        x = signal * signal

        envelope = np.zeros(shape=(num_channels, num_samples))
        # for channel in range(num_channels):
        for n in range(num_samples):
            current_envelope = np.zeros(shape=(num_channels))

            current_envelope = self.last_envelope + self.attack*np.max(x[:, n] - self.last_envelope, initial=0.0) - self.release*self.last_envelope

            self.last_envelope = current_envelope
            envelope[:, n] = current_envelope
        return envelope
        