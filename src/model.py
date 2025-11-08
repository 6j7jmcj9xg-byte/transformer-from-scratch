import torch
import torch.nn as nn
import math
import json

# ---------------------- Multi-Head Self-Attention ----------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by number of heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.query = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values).view(N, value_len, self.heads, self.head_dim).transpose(1, 2)
        keys = self.keys(keys).view(N, key_len, self.heads, self.head_dim).transpose(1, 2)
        query = self.query(query).view(N, query_len, self.heads, self.head_dim).transpose(1, 2)

        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.heads, -1, -1)  # Adjust the mask shape to match attention heads
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=-1)
        attention = self.dropout(attention)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.transpose(1, 2).contiguous().view(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out


# ---------------------- Positional Encoding ----------------------

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_expansion=4, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size * ff_expansion, embed_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.attention(x, x, x, mask)
        x = self.norm1(attention + self.dropout(x))
        forward = self.feed_forward(x)
        out = self.norm2(forward + self.dropout(x))
        return out


# ---------------------- Encoder ----------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, max_length, use_positional_encoding=True, ff_expansion=4, dropout=0.1):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.use_positional_encoding = use_positional_encoding
        self.word_embedding = nn.Embedding(vocab_size, embed_size)

        if self.use_positional_encoding:
            self.position_embedding = PositionalEncoding(embed_size, max_length)

        self.layers = nn.ModuleList([
            EncoderBlock(embed_size, heads, ff_expansion, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, x, mask):
        out = self.word_embedding(x)

        if self.use_positional_encoding:
            out = self.position_embedding(out)

        out = self.dropout(out)

        for layer in self.layers:
            out = layer(out, mask)

        out = self.layer_norm(out)
        return out


# ---------------------- Decoder ----------------------

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_expansion=4, dropout=0.1):
        super(DecoderBlock, self).__init__()

        self.self_attention = MultiHeadAttention(embed_size, heads, dropout)
        self.cross_attention = MultiHeadAttention(embed_size, heads, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size * ff_expansion, embed_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        self_attn = self.self_attention(x, x, x, trg_mask)
        x = self.norm1(self_attn + self.dropout(x))

        if enc_out is not None:
            cross_attn = self.cross_attention(enc_out, enc_out, x, src_mask)
            x = self.norm2(cross_attn + self.dropout(x))

        forward = self.feed_forward(x)
        out = self.norm3(forward + self.dropout(x))
        return out


# ---------------------- Complete Transformer ----------------------

class CompleteTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size=256, num_heads=8, num_layers=4, seq_length=128, use_positional_encoding=True, use_decoder=True, ff_expansion=4, dropout=0.1):
        super(CompleteTransformer, self).__init__()

        self.embed_size = embed_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.seq_length = seq_length

        # Encoder
        self.encoder = Encoder(vocab_size, embed_size, num_layers, num_heads, seq_length, use_positional_encoding, ff_expansion, dropout)

        # Decoder (optional)
        if use_decoder:
            self.decoder_layers = nn.ModuleList([
                DecoderBlock(embed_size, num_heads, ff_expansion, dropout)
                for _ in range(num_layers)
            ])
            self.decoder_norm = nn.LayerNorm(embed_size)

        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, mask, encoder_input=None, src_mask=None, trg_mask=None):
        enc_output = self.encoder(x, mask)
        
        if encoder_input is None:  # For using encoder-only functionality
            out = self.fc_out(enc_output)
        else:  # For using encoder-decoder functionality (i.e., with cross-attention)
            dec_output = enc_output
            for layer in self.decoder_layers:
                dec_output = layer(dec_output, enc_output, src_mask, trg_mask)
            out = self.decoder_norm(dec_output)
            out = self.fc_out(out)
        
        return out
