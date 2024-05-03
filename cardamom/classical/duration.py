from enum import Enum
from typing import Union

def samples2seconds(samples: float, sample_rate: int) -> float:
    return samples / sample_rate

def samples2ms(samples: float, sample_rate: int) -> float:
    return 1000 * samples2seconds(samples, sample_rate)

def seconds2samples(seconds: float, sample_rate: int, as_int: bool=True) -> float:
    samples = seconds / sample_rate
    if as_int:
        samples = int(samples)
    return samples
    
class DurationUnits(Enum):
    MILLISECONDS = 'ms'
    SECONDS = 's'
    SAMPLES = 'samples'
    FRAMES = 'frames'

    def str2units(s):
        for e in DurationUnits:
            if e.value == s:
                return e
        return None

class Duration():
    """
    Class to store duration and easily convert from time to samples to frames.
    """
    def __init__(self, value: float, units: Union[DurationUnits, str], *, sample_rate: int=None):
        self.value = value
        self.units = units
        self.sample_rate = sample_rate

        self.samples = None
        self.seconds = None
        self.ms = None
        self.frames = None

        if self.units == DurationUnits.SAMPLES:
            assert not self.sample_rate==None, 'if `units` is \'SAMPLES\' then `sample_rate` is required.'
            self.samples = self.value
            self.sample_rate = sample_rate
            self.seconds = samples2seconds(self.samples, self.sample_rate)
            self.ms = samples2ms(self.samples, self.sample_rate)
        elif self.units == DurationUnits.SECONDS:
            self.seconds = self.value
            self.ms = 1000.0 * self.seconds
            if self.sample_rate:
                self.samples = seconds2samples(self.seconds, self.sample_rate)
        elif self.units == DurationUnits.MILLISECONDS:
            self.ms = self.value
            self.seconds = self.ms / 1000.0
            if self.sample_rate:
                self.samples = seconds2samples(self.seconds, self.sample_rate)
        else:
            NotImplemented("TODO: add `frames` option.")

    def get_num_samples(self, sample_rate: int=None):
        if not sample_rate:
            assert self.sample_rate, "`sample_rate` required, as none is defined."
            return self.samples
        return seconds2samples(self.seconds, sample_rate)
    
    def get_seconds(self):
        return self.seconds
    
    def get_ms(self):
        return self.ms
        