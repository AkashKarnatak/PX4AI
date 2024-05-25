#!/usr/bin/env python3

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MultiheadAttention(nn.Module):
    def __init__(
        self, n_embd, n_heads, head_size, bias=False, is_causal=False, **kwargs
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_size = head_size
        self.is_causal = is_causal
        self.query = nn.Linear(n_embd, n_heads * head_size, bias=bias)
        self.key = nn.Linear(n_embd, n_heads * head_size, bias=bias)
        self.value = nn.Linear(n_embd, n_heads * head_size, bias=bias)
        self.proj = nn.Linear(n_heads * head_size, n_embd)
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            print("Using slow manual attention")
            if kwargs.get("context_len") is None:
                raise Exception("is_causal = True requires context_len to be defined")
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(kwargs["context_len"], kwargs["context_len"]))
                == 0,
            )

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        if self.flash:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        else:
            weights = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_size)
            if self.is_causal:
                weights = weights.masked_fill(self.mask[:T, :T], float("-inf"))
            weights = F.softmax(weights, dim=-1)
            out = weights @ v  # B, self.n_heads, T, self.head_size

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_size)
        out = self.proj(out)
        return out


class Block(nn.Module):
    def __init__(self, n_embd, n_heads, head_size, is_causal=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.att = MultiheadAttention(
            n_embd, n_heads, head_size, bias=False, is_causal=False, context_len=30
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))


class AnomalyDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.LayerNorm(18),
            Block(18, n_heads=2, head_size=9, is_causal=True),
            nn.Linear(18, 32),
            nn.ReLU(),
            Block(32, n_heads=4, head_size=8, is_causal=True),
            nn.Linear(32, 16),
            nn.ReLU(),
            Block(16, n_heads=4, head_size=4, is_causal=True),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            Block(8, n_heads=2, head_size=4, is_causal=True),
            nn.Linear(8, 16),
            nn.ReLU(),
            Block(16, n_heads=4, head_size=4, is_causal=True),
            nn.Linear(16, 32),
            nn.ReLU(),
            Block(32, n_heads=4, head_size=8, is_causal=True),
            nn.Linear(32, 18),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
