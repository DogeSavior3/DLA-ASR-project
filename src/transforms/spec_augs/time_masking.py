import torchaudio.transforms as T
from torch import Tensor

from .base import BaseSpecAug


class TimeMasking(BaseSpecAug):
    def __init__(self, p=0.5, time_mask_param=100):
        super().__init__(p=p)
        self._aug = T.TimeMasking(time_mask_param=time_mask_param)

    def augment(self, spec):
        return self._aug(spec)
