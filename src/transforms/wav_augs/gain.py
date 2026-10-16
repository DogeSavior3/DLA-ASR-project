import torch
import torchaudio.transforms as T
from torch import Tensor

from .base import BaseWaveAug


class Gain(BaseWaveAug):
    def __init__(self, p=0.5, min_gain_db=-6.0, max_gain_db=6.0):
        super().__init__(p=p)
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

    def __call__(self, data: Tensor):
        return super().__call__(data)

    def augment(self, wav: Tensor) -> Tensor:
        gain_db = torch.empty(()).uniform_(self.min_gain_db, self.max_gain_db).item()
        return T.Vol(gain=gain_db, gain_type="db")(wav)
