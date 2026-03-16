#!/usr/bin/env python3
"""Inference module for AutoResearch SDPA model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import math

# Import model architecture from model.py
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import AutoResearchModel

# ==========================================
# GENERATION FUNCTIONS
# ==========================================

def sample_top_k(logits, temperature=1.0, top_k=None, top_p=None):
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    if top_k is not None and top_k > 0:
        top_k = min(top_k, probs.size(-1))
        values, indices = torch.topk(probs, top_k, dim=-1)
        probs = torch.zeros_like(probs).scatter_(-1, indices, values)

    if top_p is not None and top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        probs = probs.masked_fill(indices_to_remove, 0)

    sampled = torch.multinomial(probs, 1)

    if squeeze_output:
        return sampled.squeeze(0)
    return sampled


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_token_id=None, show_progress=False):
    model.eval()
    B, T = idx.shape

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.seq_len else idx[:, -model.seq_len:]

        logits = model(idx_cond)
        logits = logits[:, -1, :]

        next_token = sample_top_k(logits, temperature, top_k, top_p)
        next_token = next_token.unsqueeze(1)

        idx = torch.cat([idx, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

        if show_progress:
            print('.', end='', flush=True)

    if show_progress:
        print()

    return idx


# ==========================================
# CHECKPOINT MANAGEMENT
# ==========================================

def save_checkpoint(model, config, metrics, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': metrics,
        'model_class': 'AutoResearchModel'
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path):
    """Load model checkpoint and return model + config.

    Handles both formats:
    - Wrapped format: {'model_state_dict': {...}, 'config': {...}, 'metrics': {...}}
    - Original repo format: Just the state_dict
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    # Check if it's a checkpoint object with metadata or just a state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Wrapped format
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config')
        metrics = checkpoint.get('metrics', {})
    else:
        # Original repo format - plain state_dict
        state_dict = checkpoint
        config = None
        metrics = {}

    # Derive config from state_dict if not provided
    if config is None:
        vocab_size = state_dict.get('token_embed.weight', torch.empty(0)).shape[0]
        d_model = state_dict.get('token_embed.weight', torch.empty(0)).shape[1]

        # Count transformer layers
        num_layers = sum(1 for k in state_dict.keys() if 'attn_norm' in k)
        depth = num_layers

        # Find head_dim from first attention layer
        head_dim = None
        for k in state_dict.keys():
            if 'k_proj.weight' in k:
                head_dim = state_dict[k].shape[1] // depth
                break

        n_heads = d_model // head_dim if head_dim else 1
        window_pattern = "SSL"
        seq_len = 256

        config = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'depth': depth,
            'head_dim': head_dim,
            'n_heads': n_heads,
            'window_pattern': window_pattern,
            'seq_len': seq_len
        }
        print(f"Inferred config from checkpoint: {config}")

    # Create model
    model = AutoResearchModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        depth=config['depth'],
        head_dim=config['head_dim'],
        n_heads=config['n_heads'],
        window_pattern=config['window_pattern'],
        seq_len=config['seq_len']
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, config, metrics


# ==========================================
# CLI INTERFACE
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Generate text from AutoResearch model')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--max-tokens', type=int, default=100, help='Tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=None, help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=None, help='Nucleus sampling')

    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint}...")
    model, config, metrics = load_checkpoint(args.checkpoint)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Model loaded: {config}")
    print(f"Metrics: {metrics}")

    print(f"Generating {args.max_tokens} tokens...")
    generated = generate(
        model,
        torch.tensor([[0]], dtype=torch.long).to(device),
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    print(f"Generated tokens: {generated.tolist()}")


if __name__ == '__main__':
    main()