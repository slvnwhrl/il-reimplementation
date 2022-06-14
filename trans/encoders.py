"""Encoder classes used by the Transducer model."""
import argparse
import math
from trans import register_component


import torch


@register_component('lstm', 'encoder')
class LSTMEncoder(torch.nn.LSTM):
    """LSTM-based encoder."""
    def __init__(self, args: argparse.Namespace):
        super().__init__(
            input_size=args.char_dim,
            hidden_size=args.enc_hidden_dim,
            num_layers=args.enc_layers,
            bidirectional=args.enc_bidirectional,
            dropout=args.enc_dropout,
            device=args.device
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--enc-hidden-dim", type=int, default=200,
                            help="Encoder LSTM state dimension.")
        parser.add_argument("--enc-layers", type=int, default=1,
                            help="Number of encoder LSTM layers.")
        parser.add_argument("--enc-bidirectional", type=bool, default=True,
                            help="If LSTM is bidirectional.")
        parser.add_argument("--enc-dropout", type=float, default=0.,
                            help="Dropout probability after each LSTM layer"
                                 "(except the last layer).")

    @property
    def output_size(self):
        return self.hidden_size * 2 if self.bidirectional else self.hidden_size


@register_component('transformer', 'encoder')
class TransformerEncoder(torch.nn.Module):
    """Transformber-based encoder."""
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.d_model = args.char_dim
        self.dropout = torch.nn.Dropout(args.enc_dropout)
        self.embed_scale = math.sqrt(args.char_dim)
        self.pos_encoding = SinusoidalPositionalEmbedding(
            embedding_dim=args.char_dim,
            padding_idx=1
        )
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=args.char_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_dim_feedforward,
            dropout=args.enc_dropout,
            norm_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=args.enc_layers,
            norm=torch.nn.LayerNorm(args.char_dim)
        )
        self.to(args.device)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        scaled_embed = self.embed_scale * src
        pos_embed = self.pos_encoding(src_key_padding_mask.int().transpose(0, 1))
        embed = self.dropout(scaled_embed + pos_embed)
        return self.transformer_encoder(embed, mask, src_key_padding_mask)

    @property
    def output_size(self):
        return self.d_model

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--enc-layers", type=int, default=4,
                            help="Number of Transformer encoder layers.")
        parser.add_argument("--enc-nhead", type=int, default=4,
                            help="Number of Transformer heads.")
        parser.add_argument("--enc-dim-feedforward", type=int, default=1024,
                            help="Number of Transformer heads.")
        parser.add_argument("--enc-dropout", type=float, default=0.1,
                            help="Dropout probability.")


class SinusoidalPositionalEmbedding(torch.nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        mask = input.ne(self.padding_idx).long()
        positions = torch.cumsum(mask, dim=0) * mask + self.padding_idx
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )

