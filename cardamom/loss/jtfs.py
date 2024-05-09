import torch
import numpy as np
from kymatio.torch import TimeFrequencyScattering
from .distance import DistanceLoss
"""
 TimeFrequencyScatteringLoss

 Joint Time-Frequency Scattering loss



Args:
    shape (tuple): The shape of the input signals in the form (num_samples, ).
    Q (tuple, optional): The quality factors for the filters at each order. Defaults to (8, 2).
    J (int, optional): The number of scales for the scattering transform. Defaults to 12.
    J_fr (int, optional): The number of scales for the frequency scattering filterbank. Defaults to 3.
    Q_fr (int, optional): The quality factor for the filters in the frequential filterbank. Defaults to 2.
    F (int, optional): Frequency averaging scale. Defgault to 2**J_fr
    T (int, optional): Temporal averaging scale. Defaults to 2**J
    format (str, optional): output format with options ["time", "joint"]
    p (float, optional): The exponent value for the distance metric. Defaults to 2.0.

Example:
    >>> loss = TimeFrequencyScatteringS2Loss(shape=(batch_size, channels, height, width), Q=(6, 3), J=10, J_fr=2, Q_fr=3)
"""


class TimeFrequencyScatteringLoss(DistanceLoss):

    def __init__(self,
                 shape,
                 Q=(8, 2),
                 J=12,
                 J_fr=3,
                 Q_fr=2,
                 F=None,
                 T=None,
                 format="time",
                 weights=(1.0, 1.0),
                 device="cpu"):
        super().__init__()

        self.shape = shape
        self.Q = Q
        self.J = J
        self.J_fr = J_fr
        self.F = F
        self.Q_fr = Q_fr
        self.T = T
        self.format = format
        self.operator_init()
        self.weights = weights
        self.device = device

    def operator_init(self):
        S = TimeFrequencyScattering(
            shape=self.shape,
            Q=self.Q,
            J=self.J,
            J_fr=self.J_fr,
            Q_fr=self.Q_fr,
            T=self.T,
            F=self.F,
            format=self.format,
        ).to(self.device)
        self.operator = [S]
        self.idxs = (np.where(S.meta()["order"] == 1),
                     np.where(S.meta()["order"] == 2))

    def forward(self, x, y, transform_y=True):
        loss = torch.tensor(0.0).type_as(x)
        for op in self.operator:
            Sx = op(x)
            Sy = op(y)[0] if transform_y else y
            for i, w in enumerate(self.weights):
                loss += w * self.cosine(Sx[self.idxs[i]], Sy[self.idxs[i]])
        loss /= len(self.operator)
        return loss
