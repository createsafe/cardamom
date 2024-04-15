import torch


class MagnitudeSTFT(torch.nn.Module):

    def __init__(self, n_fft, hop_length):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x):
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).type_as(x),
            return_complex=True,
        ).abs()
