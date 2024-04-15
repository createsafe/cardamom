import torch
from .distance import DistanceLoss
from spectral.stft import MagnitudeSTFT


class MultiScaleSpectralLoss(DistanceLoss):
    """
    Multi-scale spectral loss module.

    Args:
        max_n_fft (int, optional): The maximum size of the FFT (Fast Fourier Transform). Defaults to 2048.
        num_scales (int, optional): The number of scales to consider. Defaults to 6.
        hop_lengths (list, optional): The hop lengths for each scale. If not provided, they are computed automatically. Defaults to None.
        mag_w (float, optional): The weight for the magnitude component. Defaults to 1.0.
        logmag_w (float, optional): The weight for the log-magnitude component. Defaults to 0.0.
        p (float, optional): The exponent value for the distance metric. Defaults to 1.0.

    Notes:
        - The `max_n_fft` parameter should be divisible by 2 raised to the power of (`num_scales` - 1).
        - If `hop_lengths` are not provided, they are automatically computed based on the `n_ffts` of each scale.

    Example:
        >>> loss = MultiScaleSpectralLoss(max_n_fft=4096, num_scales=4, mag_w=0.8, logmag_w=0.2, p=2.0)
    """

    def __init__(self,
                 max_n_fft=2048,
                 num_scales=6,
                 hop_lengths=None,
                 mag_w=1.0,
                 logmag_w=0.0,
                 device="cpu"):
        super().__init__()
        assert max_n_fft // 2**(num_scales - 1) > 1
        self.max_n_fft = 2048
        self.n_ffts = [max_n_fft // (2**i) for i in range(num_scales)]
        self.hop_lengths = ([n // 4 for n in self.n_ffts]
                            if not hop_lengths else hop_lengths)
        self.mag_w = mag_w
        self.logmag_w = logmag_w

        self.device = device

        self.operator_init()

    def operator_init(self):
        self.operator = [
            MagnitudeSTFT(n_fft, self.hop_lengths[i]).to(self.device)
            for i, n_fft in enumerate(self.n_ffts)
        ]

    def forward(self, x, y):
        loss = torch.tensor(0.0).type_as(x)
        for op in self.operator:
            loss += self.cosine(op(x), op(y))
        loss /= len(self.operator)
        return loss
