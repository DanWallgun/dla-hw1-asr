from typing import List, NamedTuple, Union

import numpy as np
import torch
import sentencepiece as spm
from torch import Tensor
from pyctcdecode import build_ctcdecoder

from hw_asr.base.base_text_encoder import BaseTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class ExternalCTCBPETextEncoder(BaseTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, model_file, lm_file: str = None):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.vocab = self.sp.IdToPiece(list(range(self.sp.GetPieceSize())))
        self.decoder = build_ctcdecoder([''] + self.vocab)
    
    def __len__(self):
        return 1 + len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        if item == 0:
            return self.EMPTY_TOK
        return self.sp.DecodeIds(item - 1)
    
    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        try:
            return Tensor(self.sp.EncodeAsIds(text, enable_sampling=True, nbest_size=-1, alpha=0.1)).unsqueeze(0) + 1
        except KeyError as e:
            raise Exception(
                f"Can't encode text '{text}''")

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        """raw decode into bpe tokens"""
        return ''.join([self[int(ind)] for ind in vector]).strip()
    
    def ctc_decode(self, inds: List[int]) -> str:
        compressed = [inds[0]]
        for ind in inds[1:]:
            if compressed[-1] != ind:
                compressed.append(ind)
        empty_tok_ind = 0
        return self.sp.DecodeIds([int(ind) - 1 for ind in compressed if ind != empty_tok_ind])

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        hypos: List[Hypothesis] = []

        beams = self.decoder.decode_beams(probs[:probs_length].log().numpy(), beam_size)
        for text, _, _, logit_score, _ in beams:
            hypos.append(Hypothesis(text, logit_score))

        return sorted(hypos, key=lambda x: x.prob, reverse=True)
