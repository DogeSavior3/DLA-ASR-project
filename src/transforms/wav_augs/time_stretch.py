import librosa
import torch
from torch import Tensor

from .base import BaseWaveAug


class TimeStretch(BaseWaveAug):
    def __init__(self, p=0.5, min_rate=0.9, max_rate=1.1):
        super().__init__(p=p)
        self.min_rate = min_rate
        self.max_rate = max_rate

    def __call__(self, data: Tensor):
        return super().__call__(data)

    def augment(self, wav: Tensor) -> Tensor:
        rate = torch.empty(()).uniform_(self.min_rate, self.max_rate).item()
        stretched = librosa.effects.time_stretch(wav.detach().cpu().numpy(), rate=rate)
        return torch.from_numpy(stretched).to(wav)
