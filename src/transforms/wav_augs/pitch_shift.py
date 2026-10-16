import librosa
import torch
from torch import Tensor

from .base import BaseWaveAug


class PitchShift(BaseWaveAug):
    def __init__(self, p=0.5, sample_rate=16000, min_steps=-3, max_steps=3):
        super().__init__(p=p)
        self.sample_rate = sample_rate
        self.min_steps = min_steps
        self.max_steps = max_steps

    def __call__(self, data: Tensor):
        return super().__call__(data)

    def augment(self, wav: Tensor) -> Tensor:
        n_steps = torch.empty(()).uniform_(self.min_steps, self.max_steps).item()
        shifted = librosa.effects.pitch_shift(
            wav.detach().cpu().numpy(), sr=self.sample_rate, n_steps=n_steps
        )
        return torch.from_numpy(shifted).to(wav)
