import torch
from torch import Tensor

from .base import BaseWaveAug


class GaussianNoise(BaseWaveAug):
    def __init__(self, p=0.5, noise_std=0.05):
        super().__init__(p=p)
        self.noise_std = noise_std

    def __call__(self, data: Tensor):
        return super().__call__(data)

    def augment(self, wav: Tensor) -> Tensor:
        noise = torch.randn(wav.shape, device=wav.device, dtype=wav.dtype)
        return wav + self.noise_std * noise
