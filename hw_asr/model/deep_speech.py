import torch
from torch import nn

from hw_asr.base import BaseModel


def length_narrow(length, kernel_size, stride, padding, dilation=1):
    """According to https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html"""
    # return (length + 2 * padding - dilation * (kernel_size - 1) - 1 + stride) // stride
    return torch.div(
        length + 2 * padding - dilation * (kernel_size - 1) - 1 + stride, 
        stride, rounding_mode='trunc'
    )


class MaskedConv2d(nn.Conv2d):
    def forward(self, x, lengths):
        """
        x: Batch x Channel x Feature x Sequence
        lengths: Batch
        """
        assert lengths.dim() == 1, f'{lengths.size()=}'
        assert x.size(0) == lengths.size(0), f'{x.size()=} {lengths.size()=}'

        x = super().forward(x)
        lengths = length_narrow(lengths, self.kernel_size[1], self.stride[1], self.padding[1])
        # max_length = lengths.max()
        # padding_mask = (torch.arange(max_length, device=lengths.device).expand(lengths.size(0), max_length) >= lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(1)
        padding_mask = torch.zeros_like(x, dtype=torch.bool)
        for ind, length in enumerate(lengths):
            padding_mask[ind, :, :, length.item():].fill_(True)
        x.masked_fill_(mask=padding_mask, value=0.0)

        return x, lengths


class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.layers = nn.ModuleList([
            MaskedConv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            MaskedConv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        ])

    def get_output_size(self, input_size, dim=0):
        for layer in self.layers:
            if isinstance(layer, MaskedConv2d):
                input_size = length_narrow(
                    input_size,
                    layer.kernel_size[dim],
                    layer.stride[dim],
                    layer.padding[dim],
                )
        return input_size
        
    def forward(self, x, lengths):
        for layer in self.layers:
            if isinstance(layer, MaskedConv2d):
                x, lengths = layer(x, lengths)
            else:
                x = layer(x)
        return x, lengths


class RNNBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNBlock, self).__init__()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.act = nn.ReLU()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )

    def forward(self, x, lengths):
        # total_length = x.size(0)

        x = self.act(self.batch_norm(x.transpose(-1, -2))).transpose(-1, -2)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        return x


class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(RNNEncoder, self).__init__()
        self.layers = nn.ModuleList([
            RNNBlock(input_size=input_size, hidden_size=hidden_size),
            # RNNBlock(input_size=hidden_size * 2, hidden_size=hidden_size),  # bidirectional, so multiply by 2
        ])
        self.ln = nn.LayerNorm(hidden_size * 2)
    def forward(self, x, lengths):
        for layer in self.layers:
            x = layer(x, lengths)
        x = self.ln(x)
        return x


class KindaDeepSpeechModel(BaseModel):
    """
    Описание модели https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html
    Основной конфиг https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text/ds2_small_1gpu.py
    """
    
    def __init__(self, n_feats, n_class, **batch):
        super().__init__(n_feats, n_class, **batch)
        rnn_hidden_size = 512
        self.conv_encoder = ConvEncoder()
        self.rnn_encoder = RNNEncoder(32 * self.conv_encoder.get_output_size(n_feats, dim=0), hidden_size=rnn_hidden_size)
        self.fc = nn.Linear(rnn_hidden_size * 2, n_class)

    def forward(self, spectrogram, spectrogram_length, **batch):
        x, lengths = self.conv_encoder(spectrogram.unsqueeze(1), spectrogram_length)  # unsqueeze to add channel dimension
        x = x.permute(0, 3, 1, 2)  # Batch x Channel x Feature x Sequence -> Batch x Sequence x Channel x Feature
        x = x.view(x.size(0), x.size(1), -1)  # Batch x Sequence x Channel x Hidden -> Batch x Sequence x (Channel*Feature)
        x = self.rnn_encoder(x, lengths)
        return {"logits": self.fc(x)}

    def transform_input_lengths(self, input_lengths):
        return self.conv_encoder.get_output_size(input_lengths, dim=1)
