import json
from typing import List, Union

import numpy as np
import sentencepiece as spm
from torch import Tensor

from hw_asr.base.base_text_encoder import BaseTextEncoder


class BPETextEncoder(BaseTextEncoder):
    """"""
    def __init__(self, model_file):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.vocab = [self.sp.id_to_piece(id) for id in range(self.sp.get_piece_size())]

    def __len__(self):
        return self.sp.vocab_size()

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.sp.DecodeIds(item)[0]

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        try:
            return Tensor(self.sp.Encode(text)).unsqueeze(0)
        except KeyError as e:
            raise Exception(
                f"Can't encode text '{text}''")

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        return self.sp.DecodeIds([int(ind) for ind in vector]).strip()

