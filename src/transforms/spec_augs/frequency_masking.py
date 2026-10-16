import torchaudio.transforms as T
from torch import Tensor

from .base import BaseSpecAug


class FrequencyMasking(BaseSpecAug):
    def __init__(self, p=0.5, freq_mask_param=20):
        super().__init__(p=p)
        self._aug = T.FrequencyMasking(freq_mask_param=freq_mask_param)

    def augment(self, spec):
        return self._aug(spec)
