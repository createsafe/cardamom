import torch
"""
DistanceLoss

Superclass for all loss functions containing common kernels for distance metrics

"""


class DistanceLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def L1(self, x, y):
        """Compute the L1 Norm between two 2D tensors."""
        return torch.abs(x - y).mean()

    def L2(self, x, y):
        """Compute the L2 Norm between two 2D tensors."""
        return torch.norm(x - y, p=2.0)

    def cosine(self, x, y):
        """Compute the cosine distance between two 2D tensors."""
        sim = torch.nn.functional.cosine_similarity(x, y, dim=0)
        dist = 1 - sim
        dist /= 2
        dist = dist.mean()
        return dist

    def operator_init(*args):
        raise NotImplementedError(
            "Operator must be implemented by subclasses.")

    def forward(self, x, y):
        raise NotImplementedError(
            "Forward method must be implemented by subclasses.")
