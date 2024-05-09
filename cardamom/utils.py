import os

import soundfile as sf

AUDIO_FILETYPES = tuple(["." + filetype.lower() for filetype in list(sf._formats.keys())] + [".mp3"])

def is_audio_filetype(file):
    """
    Check if file is an audio file.
    """
    ext = os.path.splitext(file)[-1]
    return ext.lower() in AUDIO_FILETYPES