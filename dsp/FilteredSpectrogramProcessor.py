from typing import Iterable
import pprint

import librosa
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

# torch.set_printoptions(profile="full")

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
    
    num_filters = len(bins) - 2
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

def log_magnitude(spectrogram: torch.Tensor, 
                  mul: float,
                  addend: float):
    return torch.log10((spectrogram * mul) + addend)

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
                 channels: int,
                 unique_bins: bool):
        
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.freqs = freqs
        self.channels = channels

        # use double fft_size so that dims match when negative 
        # frequencies are discarded
        self._spectrogram_processor = torchaudio.transforms.Spectrogram(n_fft=self.fft_size * 2,
                                                                        hop_length=self.hop_size)
        self._fft_freqs = np.linspace(0, self.sample_rate/2, self.fft_size)
        self._bins = frequencies2bins(self.freqs, self._fft_freqs, unique_bins)
        self._filters = triangular_filter(self.channels, self._bins, self.fft_size)

    def process(self, signal: torch.Tensor):
        assert len(signal.shape) == 2, "signal must have dimensions [num_channels, num_samples]"
        spectrogram = self._spectrogram_processor(signal)
        spectrogram = spectrogram[:, :self.fft_size, :] 
        return torch.matmul(self._filters, spectrogram)
        # return spectrogram
    
# *************************************************************************
# *************************************************************************
# *************************************************************************
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (
    FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
    SpectrogramDifferenceProcessor)
from madmom.processors import ParallelProcessor, SequentialProcessor
from BeatNet.common import *


# feature extractor that extracts magnitude spectrogoram and its differences  

class LOG_SPECT(FeatureModule):
    def __init__(self, num_channels=1, sample_rate=22050, win_length=2048, hop_size=512, n_bands=[12], mode='online'):
        sig = SignalProcessor(num_channels=num_channels, win_length=win_length, sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.hop_length = hop_size
        self.num_channels = num_channels
        multi = ParallelProcessor([])
        frame_sizes = [win_length]  
        num_bands = n_bands  
        for frame_size, num_bands in zip(frame_sizes, num_bands):
            if mode == 'online' or mode == 'offline':
                frames = FramedSignalProcessor(frame_size=frame_size, hop_size=hop_size) 
            else:   # for real-time and streaming modes 
                frames = FramedSignalProcessor(frame_size=frame_size, hop_size=hop_size, num_frames=4) 
            stft = ShortTimeFourierTransformProcessor()  # caching FFT window
            filt = FilteredSpectrogramProcessor(
                num_bands=num_bands, fmin=30, fmax=17000, norm_filters=True)
            spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
            diff = SpectrogramDifferenceProcessor(
                diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
            # process each frame size with spec and diff sequentially
            # multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
            multi.append(SequentialProcessor((frames, stft, filt, spec)))
        # stack the features and processes everything sequentially
        self.pipe = SequentialProcessor((sig, multi, np.hstack))

    def process_audio(self, audio):
        feats = self.pipe(audio)
        return feats.T
# *************************************************************************
# *************************************************************************
# *************************************************************************

if __name__ == '__main__':

    filename = "80bpm.wav"
    audio, sample_rate = torchaudio.load(filename)

    sampler = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                             new_freq=22050)
    audio = sampler(audio)

    bands_per_octave = 1
    fft_size = 2048
    hop_size = 512

    log_spect = LogSpacedTriangularFilterbank(channels=audio.shape[0], 
                                              sample_rate=22050,
                                              fft_size=fft_size//2,
                                              hop_size=hop_size,
                                              freqs=log_frequencies(bands_per_octave, 30, 17000),
                                              unique_bins=True)
    result = log_spect.process(audio).abs()
    result = log_magnitude(result, 1, 1)

    spec = LOG_SPECT(n_bands=[bands_per_octave])
    feats = spec.process_audio(audio.numpy().T)

    # sig = SignalProcessor(num_channels=1, win_length=fft_size, sample_rate=sample_rate)
    # frames = FramedSignalProcessor(frame_size=fft_size, hop_size=hop_size) 
    # stft = ShortTimeFourierTransformProcessor()
    # # pipe = SequentialProcessor((sig))

    # processor = SequentialProcessor((sig, frames, stft))

    # x = processor.process(audio.numpy().T)



    pass

    fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)
    frame = 10
    axs[0].plot(feats[:, frame])
    axs[1].plot(result[0, :, frame])
    # axs[0].pcolormesh(feats)
    # axs[1].pcolormesh(result[0, :])
    plt.show()