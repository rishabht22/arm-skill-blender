import torch
from torch import nn


class CompositionHead(nn.Module):
    """A shared global-state trunk coordinates two categorical skill selections."""

    def __init__(self, dimension=48, hidden=128):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("scale", torch.ones(dimension))
        self.network = nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 8))

    def forward(self, observations):
        x = ((observations - self.mean) / self.scale).clamp(-10, 10)
        return self.network(x).reshape(-1, 2, 4)


def load_head(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    head = CompositionHead(**checkpoint["architecture"])
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    return head, checkpoint
