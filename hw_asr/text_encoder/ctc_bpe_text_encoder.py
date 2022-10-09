from typing import List, NamedTuple

import torch
from pyctcdecode import build_ctcdecoder

from .bpe_text_encoder import BPETextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class ExternalCTCBPETextEncoder(BPETextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, model_file):
        super().__init__(model_file)
        self.decoder = build_ctcdecoder([''] + self.vocab)
    
    def __len__(self):
        return 1 + len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        if item == len(self.vocab):
            return self.EMPTY_TOK
        else:
            return self.sp.Decode(item)
    
    def ctc_decode(self, inds: List[int]) -> str:
        compressed = [inds[0]]
        for ind in inds[1:]:
            if compressed[-1] != ind:
                compressed.append(ind)
        empty_tok_ind = self.sp.vocab_size
        return super().decode([ind for ind in compressed if ind != empty_tok_ind])

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
