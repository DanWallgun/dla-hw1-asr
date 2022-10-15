import torchaudio.transforms
from torch import Tensor, rand

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, rate, *args, **kwargs):
        self.rate_sampler = (lambda: rate) if isinstance(rate, float) else (lambda: rand(1) * (rate[1] - rate[0]) + rate[0])
        self._aug = torchaudio.transforms.TimeStretch(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x, self.rate_sampler).squeeze(1)
