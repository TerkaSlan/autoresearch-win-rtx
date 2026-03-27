#!/usr/bin/env python3
"""
Inference module for AutoResearch SDPA model.
Uses the EXACT model architecture from train.py for 100% compatibility with GPU checkpoints.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# ==========================================
# MODEL ARCHITECTURE (EXACT COPY FROM train.py)
# ==========================================


# Register GPTConfig in __main__ module namespace for pickle compatibility
# Checkpoints saved from train.py pickle GPTConfig as __main__.GPTConfig
# This allows pickle to find it when loading in inference.py
class _PickleCompat:
    """Compatibility class to resolve pickled GPTConfig from train.py"""
    pass

# Add GPTConfig placeholder to __main__ if it exists (for pickle resolution)
if '__main__' in sys.modules:
    main_module = sys.modules['__main__']
    if not hasattr(main_module, 'GPTConfig'):
        # Create a placeholder that will be replaced after GPTConfig is defined
        pass


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    attention_backend: str = "sdpa"
    use_activation_checkpointing: bool = False
    compute_dtype: torch.dtype = torch.bfloat16


# Register GPTConfig in __main__ for pickle compatibility with train.py checkpoints
# When train.py saves a checkpoint, GPTConfig is pickled as __main__.GPTConfig
# By adding it here, pickle can resolve the class when loading
if '__main__' in sys.modules:
    sys.modules['__main__'].GPTConfig = GPTConfig

# Allowlist GPTConfig for safe loading with weights_only=True
# This suppresses the warning about unpickling __main__.GPTConfig
try:
    torch.serialization.add_safe_globals([GPTConfig])
except (AttributeError, TypeError):
    # Older PyTorch versions may not have add_safe_globals
    pass


def norm(x):
    """RMS norm - EXACT from train.py"""
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    """Apply rotary positional embeddings - EXACT from train.py"""
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    """Multi-head attention with rotary pos encoding and value embedding gating - EXACT from train.py"""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.attention_backend = config.attention_backend
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        self._mask_cache = {}

    def _get_sdpa_mask(self, seq_len, window_size, device):
        """Get SDPA attention mask with causal and sliding window - EXACT from train.py"""
        window = window_size[0] if isinstance(window_size, tuple) else window_size
        cache_key = (seq_len, int(window), device.type, device.index)
        mask = self._mask_cache.get(cache_key)
        if mask is not None:
            return mask

        row = torch.arange(seq_len, device=device).unsqueeze(1)
        col = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = col <= row  # causal
        if window is not None and window >= 0 and window < seq_len:
            mask = mask & (col >= (row - window))
        self._mask_cache[cache_key] = mask
        return mask

    def forward(self, x, ve, cos_sin, window_size):
        """Forward pass - EXACT from train.py"""
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)  # (B, KVH, T, D)
        v = v.transpose(1, 2)  # (B, KVH, T, D)
        attn_mask = self._get_sdpa_mask(T, window_size, q.device)

        # enable_gqa was added in PyTorch 2.5+, so we need to handle older versions
        use_gqa = self.n_kv_head < self.n_head
        try:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                is_causal=False,
                enable_gqa=use_gqa,
            )
        except TypeError:
            # PyTorch < 2.5 doesn't support enable_gqa, GQA handled automatically
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                is_causal=False,
            )
        y = y.transpose(1, 2)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """MLP with ReLU^2 activation - EXACT from train.py"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU^2, not SiLU!
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer block - EXACT from train.py"""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    """GPT model with all special components - EXACT from train.py"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # For inference, we'll compute rotary embeddings on-demand to handle varied sequence lengths
        self.rotary_seq_len = config.sequence_len
        head_dim_for_rotation = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, head_dim_for_rotation, dtype=config.compute_dtype
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, device=None, dtype=torch.bfloat16):
        """Precompute rotary embeddings - EXACT from train.py"""
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (self.config.rope_base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(dtype=dtype), sin.to(dtype=dtype)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _get_rotary_for_len(self, seq_len):
        """Get rotary embeddings for a specific sequence length."""
        if seq_len <= self.cos.size(1):
            return self.cos[:, :seq_len], self.sin[:, :seq_len]
        # If we need longer sequences, recompute
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(
            seq_len, head_dim, dtype=self.config.compute_dtype
        )
        return cos, sin

    def _compute_window_sizes(self, config):
        """Compute sliding window sizes per layer - EXACT from train.py"""
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def forward(self, idx):
        """Forward pass for inference - adapted from train.py"""
        B, T = idx.size()
        cos_sin = self._get_rotary_for_len(T)

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            window_size = self.window_sizes[i]
            x = block(x, ve, cos_sin, window_size)
        x = norm(x)

        logits = self.lm_head(x).float()
        logits = self.config.softcap * torch.tanh(logits / self.config.softcap)
        return logits


# ==========================================
# GENERATION FUNCTIONS
# ==========================================


def sample_top_k(logits, temperature=1.0, top_k=None, top_p=None):
    """Sample from logits with optional temperature, top-k, and top-p filtering."""
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
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        probs = probs.masked_fill(indices_to_remove, 0)

    sampled = torch.multinomial(probs, 1)

    if squeeze_output:
        return sampled.squeeze(0)
    return sampled


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_token_id=None, show_progress=False):
    """Generate text tokens - autoregressive decoding.

    Args:
        model: The GPT model
        idx: Starting token tensor (B, T)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more focused)
        top_k: Top-k sampling (limit to top K tokens)
        top_p: Nucleus sampling (cumulative probability threshold)
        eos_token_id: Optional EOS token ID for early stopping
        show_progress: Whether to print progress dots

    Returns:
        Generated token tensor (B, T + new_tokens)
    """
    # Default to BOS token as EOS (token 8188 = <|reserved_0|>)
    # This prevents the model from starting a new sequence mid-generation
    if eos_token_id is None:
        eos_token_id = 8188  # <|reserved_0|> is our BOS/EOS token

    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.sequence_len else idx[:, -model.config.sequence_len:]

        logits = model(idx_cond)
        logits = logits[:, -1, :]

        next_token = sample_top_k(logits, temperature, top_k, top_p)
        # sample_top_k already returns (B, 1) shape, no need to unsqueeze

        idx = torch.cat([idx, next_token], dim=1)

        # Stop if we hit EOS token or BOS token (indicating new sequence start)
        if (next_token == eos_token_id).all():
            break

        # Stop if we output BOS token (it means the model started a new sequence)
        # This prevents the repetitive pattern where model outputs <|reserved_0|> mid-text
        if (next_token == 8188).all():  # <|reserved_0|>
            break

        if show_progress:
            print('.', end='', flush=True)

    if show_progress:
        print()

    return idx


# ==========================================
# CHECKPOINT MANAGEMENT
# ==========================================


def _get_default_config_from_checkpoint(state_dict):
    """Derive GPTConfig from checkpoint state dict when config is not saved."""
    # Extract dimensions from wte
    vocab_size = state_dict['transformer.wte.weight'].shape[0]
    n_embd = state_dict['transformer.wte.weight'].shape[1]

    # Count layers by looking at resid_lambdas
    n_layer = state_dict['resid_lambdas'].shape[0]

    # Infer n_head and n_kv_head from attention projection shapes
    # c_q: (n_embd, n_head * head_dim) where n_head * head_dim = n_embd
    # c_k: (n_embd, n_kv_head * head_dim)
    # c_v: (n_embd, n_kv_head * head_dim)
    # ve_gate: (n_kv_head, ve_gate_channels) - most reliable way to get n_kv_head

    c_q_shape = state_dict['transformer.h.0.attn.c_q.weight'].shape
    # c_q output dim = n_head * head_dim = n_embd (usually)
    n_head_times_head_dim = c_q_shape[1]

    # Try to get n_kv_head from ve_gate shape (most reliable)
    # ve_gate layers exist in alternating layers: 1, 3, 5, ... for n_layer=8
    n_kv_head = None
    for i in range(n_layer):
        ve_gate_key = f'transformer.h.{i}.attn.ve_gate.weight'
        if ve_gate_key in state_dict:
            ve_gate_shape = state_dict[ve_gate_key].shape
            n_kv_head = ve_gate_shape[0]  # First dim is n_kv_head
            break

    # Fallback: infer from c_k shape if no ve_gate
    if n_kv_head is None:
        c_k_shape = state_dict['transformer.h.0.attn.c_k.weight'].shape
        # Assume head_dim = n_embd // n_head, and n_kv_head * head_dim = c_k_shape[1]
        # Try common head_dim values (64, 128, 256)
        for possible_head_dim in [64, 128, 256, 32]:
            if n_embd % possible_head_dim == 0:
                possible_n_head = n_embd // possible_head_dim
                if c_k_shape[1] % possible_head_dim == 0:
                    possible_n_kv_head = c_k_shape[1] // possible_head_dim
                    if possible_n_kv_head <= possible_n_head and possible_n_head % possible_n_kv_head == 0:
                        n_kv_head = possible_n_kv_head
                        break

    # Default to n_kv_head = n_head (no GQA) if still unknown
    if n_kv_head is None:
        # Assume standard multi-head (no GQA)
        # Infer head_dim from common values
        for possible_head_dim in [64, 128, 256, 32]:
            if n_embd % possible_head_dim == 0:
                n_kv_head = n_embd // possible_head_dim
                break
        if n_kv_head is None:
            n_kv_head = n_embd // 64  # Default head_dim = 64

    # Now calculate n_head
    # n_head * head_dim = n_embd, and we need to find head_dim
    # head_dim should divide both n_embd and n_kv_head * head_dim (= c_k output dim)
    # Since n_kv_head * head_dim may equal n_embd (when n_kv_head = n_head),
    # we use: head_dim = n_embd // n_head
    # But we know n_kv_head, so: head_dim = (n_kv_head * head_dim) / n_kv_head
    # From c_k: output_dim = n_kv_head * head_dim
    c_k_shape = state_dict['transformer.h.0.attn.c_k.weight'].shape
    n_kv_head_times_head_dim = c_k_shape[1]
    head_dim = n_kv_head_times_head_dim // n_kv_head
    n_head = n_embd // head_dim

    # Check for value embeddings
    has_value_embeds = any(f'value_embeds.{i}' in state_dict for i in range(n_layer))

    # Default values
    sequence_len = 2048
    window_pattern = "SSSL"
    softcap = 15.0
    rope_base = 10000

    # For CPU inference, use float32 for precision (GPU may be bf16/fp16)
    # The weights will be cast appropriately on load
    compute_dtype = torch.float32

    return GPTConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        window_pattern=window_pattern,
        attention_backend="sdpa",
        use_activation_checkpointing=False,
        compute_dtype=compute_dtype,
        softcap=softcap,
        rope_base=rope_base,
    )


def load_checkpoint(path, device: Optional[str] = None) -> Tuple[GPT, dict, dict]:
    """Load model checkpoint and return model, config, and metrics.

    Args:
        path: Path to checkpoint file (.pt or .pth)
        device: Device to load model on ('cpu' or 'cuda'). Defaults to CPU for inference.

    Returns:
        Tuple of (model, config_dict, metrics_dict)

    Handles multiple checkpoint formats:
    1. New format: {'model_state_dict': {...}, 'config': {...}, 'metrics': {...}}
    2. Train.py format: {'model_state_dict': state_dict, 'config': GPTConfig, ...}
    3. Plain state_dict: just the model state dict
    """
    if device is None:
        device = 'cpu'

    # Load checkpoint with weights_only=False to support pickled GPTConfig from train.py
    # GPTConfig is registered in __main__ module for pickle resolution compatibility
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {e}")

    config_dict = {}
    metrics_dict = {}

    # Determine checkpoint format
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # Wrapped format
            state_dict = checkpoint['model_state_dict']

            # Handle config - with weights_only=True, config may be loaded as raw dict
            # even if it was saved as GPTConfig object
            if 'config' in checkpoint:
                checkpoint_config = checkpoint['config']
                try:
                    if isinstance(checkpoint_config, dict):
                        config_dict = checkpoint_config
                    elif hasattr(checkpoint_config, '__dict__'):
                        config_dict = checkpoint_config.__dict__
                    else:
                        config_dict = dict(checkpoint_config)
                except Exception as e:
                    print(f"Warning: Could not extract config ({e}), inferring from state dict")
                    config_dict = _get_default_config_from_checkpoint(state_dict).__dict__
            else:
                config_dict = _get_default_config_from_checkpoint(state_dict).__dict__

            # Metrics
            metrics_dict = checkpoint.get('metrics', {})

            # Additional metadata - override config with checkpoint values if available
            if 'timestamp' in checkpoint:
                metrics_dict['timestamp'] = checkpoint['timestamp']
            if 'val_bpb' in checkpoint:
                metrics_dict['val_bpb'] = checkpoint['val_bpb']
            if 'step' in checkpoint:
                metrics_dict['step'] = checkpoint['step']
            if 'softcap' in checkpoint:
                config_dict['softcap'] = checkpoint['softcap']
            if 'rope_base' in checkpoint:
                config_dict['rope_base'] = checkpoint['rope_base']
            if 'window_pattern' in checkpoint:
                config_dict['window_pattern'] = checkpoint['window_pattern']

        else:
            # Plain state dict
            state_dict = checkpoint
            config_dict = _get_default_config_from_checkpoint(state_dict).__dict__
    else:
        # Unexpected format
        raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

    # Convert config dict to GPTConfig
    config = GPTConfig(**config_dict)

    # Create model
    model = GPT(config)
    model.load_state_dict(state_dict)
    model.eval()

    # Move to specified device
    model = model.to(device)

    return model, config_dict, metrics_dict


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
    parser.add_argument('--device', default='cpu', help='Device to run on (cpu/cuda)')

    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint}...")
    model, config, metrics = load_checkpoint(args.checkpoint, device=args.device)
    print(f"Model loaded with config: {config}")
    print(f"Metrics: {metrics}")

    print(f"Generating {args.max_tokens} tokens...")
    generated = generate(
        model,
        torch.tensor([[0]], dtype=torch.long).to(args.device),
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        show_progress=True
    )

    print(f"\nGenerated tokens: {generated.tolist()[0]}")


if __name__ == '__main__':
    main()