from collections import defaultdict
from typing import List, NamedTuple

import torch
from pyctcdecode import build_ctcdecoder

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        compressed = [self.ind2char[inds[0]]]
        for ind in inds[1:]:
            char = self.ind2char[ind]
            if compressed[-1] != char:
                compressed.append(char)
        return ''.join([char for char in compressed if char != self.EMPTY_TOK])

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []

        def extend_and_merge(next_char_probs, src_paths):
            new_paths = defaultdict(float)
            for next_char_ind, next_char_prob in enumerate(next_char_probs):
                next_char = self.ind2char[next_char_ind]
                for (text, last_char), path_prob in src_paths.items():
                    new_prefix = text if next_char == last_char else (text + next_char)
                    new_prefix = new_prefix.replace(self.EMPTY_TOK, '')
                    new_paths[(new_prefix, next_char)] += path_prob * next_char_prob
            return new_paths
        
        def truncate_beam(paths, beam_size):
            return dict(sorted(paths.items(), key=lambda x: x[1])[-beam_size:])

        paths = {('', self.EMPTY_TOK): 1.0}
        for probs_ind in range(probs_length):
            paths = extend_and_merge(probs[probs_ind], paths)
            paths = truncate_beam(paths, beam_size)
        
        hypos = [Hypothesis(text, path_prob) for (text, _), path_prob in paths.items()]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)


class ExternalCTCCharTextEncoder(CTCCharTextEncoder):
    """https://pypi.org/project/pyctcdecode/"""

    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, lm_file: str = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.decoder = build_ctcdecoder([''] + list(self.alphabet), kenlm_model_path=lm_file)

    def ctc_decode(self, inds: List[int]) -> str:
        compressed = [self.ind2char[inds[0]]]
        for ind in inds[1:]:
            char = self.ind2char[ind]
            if compressed[-1] != char:
                compressed.append(char)
        return ''.join([char for char in compressed if char != self.EMPTY_TOK])

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []

        beams = self.decoder.decode_beams(probs[:probs_length].log().numpy(), beam_size)
        for text, _, _, logit_score, _ in beams:
            hypos.append(Hypothesis(text, logit_score))

        return sorted(hypos, key=lambda x: x.prob, reverse=True)
