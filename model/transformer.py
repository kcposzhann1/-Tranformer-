import torch
import math
import torch.nn as nn
from torch.nn import functional as F

class Transformer(nn.Module):
    def __init__(self, embed_size, ffn_hidden_size, num_heads, num_layers, vocab_size, max_len, dropout):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = self.get_positional_encoding(embed_size, max_len)
        self.layers = nn.ModuleList([
            TransformerLayer(embed_size, ffn_hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def get_positional_encoding(self, d_model, max_len):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        position = self.pos_embedding[:, :x.size(1), :].to(x.device)
        x += position
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.fc_out(x)
        return x




class TransformerLayer(nn.Module):
    def __init__(self, embed_size, ffn_hidden_size, num_heads, dropout):
        super(TransformerLayer, self).__init__()
        self.norm_1 = nn.LayerNorm(embed_size)
        self.norm_2 = nn.LayerNorm(embed_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, ffn_hidden_size),
            nn.ReLU(),
            nn.Linear(ffn_hidden_size, embed_size)
        )

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm_1(x + self.dropout_1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm_2(x + self.dropout_2(ffn_out))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([
            SelfAttention(embed_size, embed_size // num_heads)
            for _ in range(num_heads)
        ])
        self.linear = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.linear(out)

class SelfAttention(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(SelfAttention, self).__init__()
        self.W_Q = nn.Linear(input_dim, embed_dim)
        self.W_K = nn.Linear(input_dim, embed_dim)
        self.W_V = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)
        attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        attn_probs = F.softmax(attn_scores, dim=-1)
        return attn_probs @ V

class ChatTransformer(nn.Module):
    def __init__(self, embed_size, ffn_hidden_size, num_heads, num_layers, vocab_size, max_seq_len, dropout):
        super(ChatTransformer, self).__init__()
        self.transformer = Transformer(
            embed_size,
            ffn_hidden_size,
            num_heads,
            num_layers,
            vocab_size,
            max_seq_len,
            dropout
        )

    def forward(self, x):
        return self.transformer(x)
