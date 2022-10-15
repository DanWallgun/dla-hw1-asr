import torchaudio.transforms
from torch import Tensor
from numpy.random import uniform

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, rate, *args, **kwargs):
        self.rate_sampler = (lambda: rate) if isinstance(rate, float) else (lambda: uniform(*rate))
        self._aug = torchaudio.transforms.TimeStretch(*args, **kwargs)

    def __call__(self, data: Tensor):
        dtype = data.dtype
        rate = self.rate_sampler()
        x = data.unsqueeze(1)
        return self._aug(x, rate).squeeze(1).type(dtype)
