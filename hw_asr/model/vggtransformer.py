import torch
from torch import nn

from hw_asr.base import BaseModel
from .deep_speech import length_narrow


"""
https://arxiv.org/abs/1904.11660
implementation inspired by https://github.com/facebookresearch/fairseq
"""

DEFAULT_ENC_VGGBLOCK_CONFIG = ((32, 3, 2, 2, False),) * 2
DEFAULT_ENC_TRANSFORMER_CONFIG = ((256, 4, 1024, True, 0.2, 0.2, 0.2),) * 4
# 256: embedding dimension
# 4: number of heads
# 1024: FFN
# True: apply layerNorm before (dropout + resiaul) instead of after
# 0.2 (dropout): dropout after MultiheadAttention and second FC
# 0.2 (attention_dropout): dropout in MultiheadAttention
# 0.2 (relu_dropout): dropout after ReLu
DEFAULT_DEC_TRANSFORMER_CONFIG = ((256, 2, 1024, True, 0.2, 0.2, 0.2),) * 2
DEFAULT_DEC_CONV_CONFIG = ((256, 3, True),) * 2


class VGGBlock(nn.Module):
    def __init__(self, config, in_ch, feature_dim):
        super().__init__()
        out_ch, kernel_size, pool_ks, num_layers, ln = config
        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            self.layers.append(nn.Conv2d(
                in_ch if idx == 0 else out_ch,
                out_ch,
                kernel_size,
                padding=kernel_size // 2
            ))
            if ln:
                assert feature_dim == length_narrow(feature_dim, kernel_size, 1, kernel_size // 2)
                self.layers.append(nn.LayerNorm(feature_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(pool_ks, ceil_mode=True))
        self.output_feature_dim = length_narrow(feature_dim, pool_ks, pool_ks, 0)

    # def transform_input_lengths(self, input_lengths):
    #     for layer in self.layers:
    #         if hasattr(layer, 'kernel_size'):
    #             input_lengths = length_narrow(input_lengths, layer.kernel_size[0], layer.stride[0], layer.padding[0], layer.dilation[0])
    #     return input_lengths

    def forward(self, x):
        for layer in self.layers:
            # print(layer._get_name(), x.size())
            x = layer(x)
            # print(x.size())
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim, num_heads, fc_dim, _, dropout, attn_dropout, act_dropout = config
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, attn_dropout, batch_first=True)
        self.self_attn_layernorm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.activation_dropout = nn.Dropout(act_dropout)
        self.fc1 = nn.Linear(embed_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, embed_dim)
        self.fc_layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x, padding_mask, attn_mask=None):
        # MHSA
        residual = x
        x, _ = self.self_attn(x, x, x, key_padding_mask=padding_mask, need_weights=False, attn_mask=attn_mask)
        x = residual + self.dropout(x)
        x = self.self_attn_layernorm(x)

        # FF
        residual = x
        x = self.fc1(x)
        x = self.activation_dropout(self.activation(x))
        x = self.fc2(x)
        x = residual + self.dropout(x)
        x = self.fc_layernorm(x)
        
        return x


class VGGTransformerEncoder(nn.Module):
    def __init__(
        self,
        feature_dim,
        enc_vggblock_config=DEFAULT_ENC_VGGBLOCK_CONFIG,
        env_transformer_config=DEFAULT_ENC_TRANSFORMER_CONFIG
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_ch = 1
        self.pooling_kernel_sizes = []
        for vgg_config in enc_vggblock_config:
            self.conv_layers.append(VGGBlock(vgg_config, in_ch, feature_dim))
            in_ch = vgg_config[0]
            feature_dim = self.conv_layers[-1].output_feature_dim
            self.pooling_kernel_sizes.append(vgg_config[2])
        transformer_input_dim = in_ch * feature_dim
        # print('transformer_input_dim', transformer_input_dim)
        self.transformer_layers = nn.ModuleList()
        self.transformer_layers.append(nn.Linear(transformer_input_dim, env_transformer_config[0][0]))
        for transformer_config in env_transformer_config:
            self.transformer_layers.append(TransformerEncoderLayer(transformer_config))

        # self.output_dim = output_dim
        self.output_dim = transformer_config[0]
        self.transformer_layers.extend([
            # Linear(transformer_config[0], output_dim),
            nn.LayerNorm(self.output_dim),
        ])

    def forward(self, x, lengths, **kwargs):
        # x - Batch x Feature x Sequence
        x = x.transpose(-1, -2).unsqueeze(1)  # x - Batch x Channel(1) x Sequence x Feature

        for layer in self.conv_layers:
            x = layer(x)
        lengths = self.transform_input_lengths(lengths)

        x = x.transpose(1, 2)  # x - Batch x Sequence x Channel x Feature
        x = x.reshape(x.size(0), x.size(1), -1)  # x - Batch x Sequence x (Channel*Feature)

        padding_mask = torch.zeros(x.size()[:2], dtype=torch.bool, device=x.device)
        for ind, length in enumerate(lengths):
            padding_mask[ind, length.item():].fill_(True)
        
        for layer_idx in range(len(self.transformer_layers)):
            if isinstance(self.transformer_layers[layer_idx], TransformerEncoderLayer):
                x = self.transformer_layers[layer_idx](
                    x, padding_mask
                )
            else:
                x = self.transformer_layers[layer_idx](x)

        return x

    def transform_input_lengths(self, input_lengths):
        # for layer in self.conv_layers:
        #     input_lengths = layer.transform_input_lengths(input_lengths)
        # return input_lengths
        for pool_ks in self.pooling_kernel_sizes:
            input_lengths = (input_lengths.float() / pool_ks).ceil().long()
        return input_lengths


class TransformerDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()


class VGGTransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()


class VGGTransformerModel(BaseModel):
    def __init__(self, n_feats, n_class, **kwargs):
        super().__init__(n_feats, n_class, **kwargs)
        self.encoder = VGGTransformerEncoder(n_feats)
        self.fc = nn.Linear(self.encoder.output_dim, n_class)

    def forward(self, spectrogram, spectrogram_length, **batch):
        x = self.encoder(x=spectrogram, lengths=spectrogram_length, **batch)
        x = self.fc(x)
        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return self.encoder.transform_input_lengths(input_lengths)
