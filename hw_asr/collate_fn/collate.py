import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    result_batch['audio'] = torch.nn.utils.rnn.pad_sequence(
        [item['audio'].squeeze(0) for item in dataset_items],
        batch_first=True
    )
    result_batch['spectrogram'] = torch.nn.utils.rnn.pad_sequence(
        [item['spectrogram'].squeeze(0).transpose(-1, -2) for item in dataset_items],
        batch_first=True
    ).transpose(-1, -2)
    result_batch['spectrogram_length'] = torch.LongTensor([item['spectrogram'].squeeze(0).size(-1) for item in dataset_items])
    result_batch['duration'] = [item['duration'] for item in dataset_items]
    result_batch['text'] = [item['text'] for item in dataset_items]
    result_batch['text_encoded'] = torch.nn.utils.rnn.pad_sequence(
        [item['text_encoded'].squeeze(0) for item in dataset_items],
        batch_first=True
    )
    result_batch['text_encoded_length'] = torch.LongTensor([item['text_encoded'].size(1) for item in dataset_items])
    result_batch['audio_path'] = [item['audio_path'] for item in dataset_items]
    return result_batch