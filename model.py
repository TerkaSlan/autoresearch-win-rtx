#!/usr/bin/env python3
"""
Model architecture for AutoResearch SDPA.
Standalone module with minimal dependencies for inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# MODEL ARCHITECTURE
# ==========================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, freqs):
    q_embed = (q * freqs.cos()) + (rotate_half(q) * freqs.sin())
    k_embed = (k * freqs.cos()) + (rotate_half(k) * freqs.sin())
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, d_model, head_dim, n_heads, window_pattern="SSL", rope=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_model = d_model
        self.window_pattern = window_pattern
        self.rope = rope

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        if rope:
            self.rotary_emb = RotaryEmbedding(head_dim)

    def create_sliding_window_mask(self, seq_len, window_size):
        mask = torch.ones(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            mask[i, start:end] = 0
        return mask.masked_fill(mask.bool(), float('-inf'))

    def forward(self, x):
        B, T, C = x.shape

        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rope:
            freqs = self.rotary_emb(T)
            Q, K = apply_rotary_pos_emb(Q, K, freqs)

        if self.window_pattern.startswith("S"):
            window_size = T // 4 if self.window_pattern == "SSL" else T // 8
            mask = self.create_sliding_window_mask(T, window_size)
            attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = attn + mask.unsqueeze(0).unsqueeze(0)
        else:
            attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(B, T, -1)

        return self.out_proj(output)

class MLP(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate(x))
        up = self.up(x)
        return self.down(gate * up)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, head_dim, n_heads, window_pattern="SSL", rope=True):
        super().__init__()
        self.attn = Attention(d_model, head_dim, n_heads, window_pattern, rope)
        self.mlp = MLP(d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.mlp_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

class AutoResearchModel(nn.Module):
    def __init__(self, vocab_size, d_model, depth, head_dim, n_heads, window_pattern="SSL", seq_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.depth = depth
        self.seq_len = seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, head_dim, n_heads, window_pattern)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embed(idx)
        pos = torch.arange(T).unsqueeze(0)
        pos_emb = self.pos_embed(pos)
        x = tok_emb + pos_emb

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits