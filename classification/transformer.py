# We adapt all layers and functions defined in EncoderBlock from https://d2l.ai/chapter_attention-mechanisms/transformer.html

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


def transpose_qkv(x, num_heads):
    """
    Transposition for parallel computation of multiple attention heads.
    :param x: (b, n_pts, c)
    :param num_heads: a scalar
    :return: (b * num_heads, n_pts, c / num_heads)
    """
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    x = x.permute(0, 2, 1, 3).contiguous()  # (b, num_heads, n_pts, c / num_heads)
    out = x.reshape(-1, x.shape[2], x.shape[3])

    return out


def transpose_output(x, num_heads):
    """
    Reverse the operation of transpose_qkv.
    :param x: (b * n_heads, n_pts, c / n_heads)
    :param num_heads: a scalar
    :return: (b, n_pts, c)
    """
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
    x = x.permute(0, 2, 1, 3).contiguous()  # (b, n_pts, num_heads, c)
    out = x.reshape(x.shape[0], x.shape[1], -1)

    return out


class DotProductAttention(nn.Module):
    """Scaled dot product attention."""

    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask=None):
        """
        :param queries: (b * n_heads, n_pts, c / n_heads)
        :param keys: (b * n_heads, n_pts / 2, c / n_heads)
        :param values: (b * n_heads, n_pts / 2, c / n_heads)
        :return: same as queries
        """
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2).contiguous()) / math.sqrt(d)  # (b * n_heads, n_pts, n_pts / 2)

        if mask is not None:
            scores = torch.masked_fill(scores, ~mask.bool(), -1e6)

        self.attention_weights = F.softmax(scores, dim=-1)

        out = torch.bmm(self.dropout(self.attention_weights), values)  # (b * n_heads, n_pts, c / n_heads)

        return out


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""

    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads

        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, mask=None):
        """
        :param queries: (b, n_pts, c)
        :param keys: (b, n_pts / 2,  c)
        :param values: (b, n_pts / 2, c)
        :return:
        """
        queries = transpose_qkv(self.W_q(queries), self.num_heads)  # (b, n_pts, c) -> (b * n_heads, n_pts, c / n_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)  # (b * n_heads, n_pts / 2, c / n_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)  # (b * n_heads, n_pts / 2, c / n_heads)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            mask = mask.reshape(-1, mask.shape[2], mask.shape[3])

        output = self.attention(queries, keys, values, mask)  # (b * n_heads, n_pts, c / n_heads)
        output_concat = transpose_output(output, self.num_heads)  # (b, n_pts, c)

        return output_concat


class PositionWiseFFN(nn.Module):
    """Positionwise feed-forward network."""

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, x):
        """
        :param x: (b, n_pts, c)
        :return: (b, n_pts, c)
        """
        return self.dense2(self.relu(self.dense1(x)))


class AddNorm(nn.Module):
    """Residual connection followed by layer normalization."""

    def __init__(self, normalized_shape, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x, y):
        return self.ln(self.dropout(y) + x)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(EncoderBlock, self).__init__()

        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias)

        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, x, context=None, mask=None):
        """
        :param x: (b, n_pts, c)
        :param context: (b, n_pts / 2, c)
        :return: (b, n_pts, c)
        """
        if context is None:
            context = x

        y = self.addnorm1(x, self.attention(x, context, context, mask))

        out = self.addnorm2(y, self.ffn(y))

        return out


class TransformerEncoder(nn.Module):
    """Transformer encoder."""

    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, bias=False):
        super(TransformerEncoder, self).__init__()

        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module('block' + str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                                              ffn_num_input, ffn_num_hiddens, num_heads, dropout, bias))

    def forward(self, x, context=None, mask=None):
        """
        :param x: (b, c, n_pts)
        :param context: (b, c, n_pts / 2)
        :return: (b, c, n_pts)
        """
        x = x.transpose(1, 2).contiguous()  # (b, n_pts, c)
        if context is not None:
            context = context.transpose(1, 2).contiguous()

        for i, blk in enumerate(self.blks):
            x = blk(x, context, mask)  # shape kept

        return x.transpose(1, 2).contiguous()


if __name__ == '__main__':
    # dummy variables
    x = torch.randn(32, 64, 1024).cuda()  # (b, c, n_pts)
    xs = torch.randn(32, 64, 512).cuda()

    key_size, query_size, value_size, num_heads = 64, 64, 64, 2  # dim for all heads
    ffn_num_input, ffn_num_hiddens, num_hiddens, num_layers, dropout = 64, 64, 64, 2, 0.1
    norm_shape = [64]

    fast_att = True
    encoder = TransformerEncoder(key_size, query_size, value_size, num_hiddens, norm_shape,
                                 ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)

    encoder = encoder.cuda()

    y = encoder(x.transpose(2, 1), xs.transpose(2, 1)).transpose(2, 1)

    print('Done.')

