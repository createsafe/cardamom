from typing import Iterable
import pprint

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(profile="full")

def log_frequencies(bands_per_octave: int, fmin: float, fmax: float, fref: float=440):
    """
    Returns frequencies aligned on a logarithmic frequency scale.

    Parameters
    ----------
    bands_per_octave : int
        Number of filter bands per octave.
    fmin : float
        Minimum frequency [Hz].
    fmax : float
        Maximum frequency [Hz].
    fref : float, optional
        Tuning frequency [Hz].

    Returns
    -------
    log_frequencies : numpy array
        Logarithmically spaced frequencies [Hz].

    Notes
    -----
    If `bands_per_octave` = 12 and `fref` = 440 are used, the frequencies are
    equivalent to MIDI notes.

    """
    # get the range
    left = np.floor(np.log2(float(fmin) / fref) * bands_per_octave)
    right = np.ceil(np.log2(float(fmax) / fref) * bands_per_octave)
    # generate frequencies
    frequencies = fref * 2. ** (torch.arange(left, right) /
                                float(bands_per_octave))
    # filter frequencies
    # needed, because range might be bigger because of the use of floor/ceil
    frequencies = frequencies[torch.searchsorted(frequencies, fmin):]
    frequencies = frequencies[:torch.searchsorted(frequencies, fmax, right=True)]
    # return
    return frequencies

def frequencies2bins(frequencies, bin_frequencies, unique_bins=False):
    """
    Map frequencies to the closest corresponding bins.

    Parameters
    ----------
    frequencies : numpy array
        Input frequencies [Hz].
    bin_frequencies : numpy array
        Frequencies of the (FFT) bins [Hz].
    unique_bins : bool, optional
        Return only unique bins, i.e. remove all duplicate bins resulting from
        insufficient resolution at low frequencies.

    Returns
    -------
    bins : numpy array
        Corresponding (unique) bins.

    Notes
    -----
    It can be important to return only unique bins, otherwise the lower
    frequency bins can be given too much weight if all bins are simply summed
    up (as in the spectral flux onset detection).

    """
    # cast as numpy arrays
    frequencies = np.asarray(frequencies)
    bin_frequencies = np.asarray(bin_frequencies)
    # map the frequencies to the closest bins
    # solution found at: http://stackoverflow.com/questions/8914491/
    indices = bin_frequencies.searchsorted(frequencies)
    indices = np.clip(indices, 1, len(bin_frequencies) - 1)
    left = bin_frequencies[indices - 1]
    right = bin_frequencies[indices]
    indices -= frequencies - left < right - frequencies
    # only keep unique bins if requested
    if unique_bins:
        indices = np.unique(indices)
    # return the (unique) bin indices of the closest matches
    return indices

def triangular_filter(channels, bins, fft_size, overlap=True, normalize=True):
    # TODO: add num channels argument 
    num_filters = len(bins) - 3
    filters = torch.zeros(size=[num_filters, fft_size])

    for n in range(num_filters):
        # get start, center and stop bins
        start, center, stop = bins[n:n+3]
        
        if not overlap:
            start = int(np.floor((center + start)) / 2)
            stop = int(np.ceil((center + stop)) / 2)

        if stop - start < 2:
            center = start
            stop = start + 1

        filters[n, start:center] = torch.linspace(start=0, end=(1 - (1 / (center-start))), steps=center-start)
        filters[n, center:stop] = torch.linspace(start=1, end=(0 + (1 / (center-start))), steps=stop-center)

    if normalize:
        filters = torch.div(filters.T, filters.sum(dim=1)).T

    filters = filters.repeat(channels, 1, 1)

    return filters    

class LogSpacedTriangularFilterbank():
    """

    Example: 
        audio, sample_rate = torchaudio.load("some/audio.wav")
        log_spect = LogSpacedTriangularFilterbank(channels=2, sample_rate=sample_rate, freqs=log_frequencies(12, 40, sample_rate/2))
        result = log_spect.process(audio)
    """
    def __init__(self, *,
                 sample_rate: int=48000, 
                 fft_size: int=4096,
                 hop_size: int=1024,
                 freqs: Iterable[float],
                 channels: int
                 ):
        
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.freqs = freqs
        self.channels = channels

        # use double fft_size so that dims match when negative 
        # frequencies are discarded
        self._spectrogram_processor = torchaudio.transforms.Spectrogram(n_fft=self.fft_size*2,
                                                      hop_length=self.hop_size)
        self._fft_freqs = np.linspace(0, self.sample_rate/2, self.fft_size)
        self._bins = frequencies2bins(self.freqs, self._fft_freqs)
        self._filters = triangular_filter(self.channels, self._bins, self.fft_size)

    def process(self, signal: torch.Tensor):
        assert len(signal.shape) == 2, "signal must have dimensions [num_channels, num_samples]"
        spectrogram = self._spectrogram_processor(signal)
        spectrogram = spectrogram[:, :self.fft_size, :] 
        return torch.matmul(self._filters, spectrogram)

audio, sample_rate = torchaudio.load("80bpm.wav")
log_spect = LogSpacedTriangularFilterbank(channels=2, sample_rate=sample_rate, freqs=log_frequencies(12, 40, sample_rate/2))
result = log_spect.process(audio)

def log_magnitude(spectrogram: torch.Tensor, 
                  mul: float,
                  addend: float):
    return torch.log10((spectrogram * mul) + addend)



result = log_magnitude(result, 1, 1)

plt.pcolormesh(result[0, :])
plt.show()