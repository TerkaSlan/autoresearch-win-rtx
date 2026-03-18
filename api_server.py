#!/usr/bin/env python3
"""
FastAPI server for AutoResearch SDPA model inference.

Features:
- CPU-only inference for GPU-trained checkpoints
- /health endpoint - check server and model status
- /generate endpoint - generate text (returns both tokens and decoded text)
- Automatic checkpoint loading from environment or file
"""

import os
import pickle
import torch
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager

# Import FastAPI and related components
try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available")
    raise ImportError("FastAPI is required. Install with: uv pip install fastapi uvicorn")

# Import inference module
from inference import load_checkpoint, generate

# Checkpoint path from environment or default
DEFAULT_CHECKPOINT = os.environ.get(
    "AUTORESEARCH_CHECKPOINT",
    os.environ.get("DEFAULT_CHECKPOINT", "checkpoints/checkpoint_best.pt")
)

# ==========================================
# Tokenizer Loading (from prepare.py cache)
# ==========================================


def _default_cache_dir():
    """Get the default cache directory (same as prepare.py)."""
    env_cache = os.environ.get("AUTORESEARCH_CACHE_DIR")
    if env_cache:
        return os.path.expanduser(env_cache)

    legacy_cache = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
    if os.name != "nt":
        return legacy_cache

    if os.path.exists(legacy_cache):
        return legacy_cache

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return os.path.join(local_app_data, "autoresearch")
    return legacy_cache


def _resolve_dataset_name(dataset_name: Optional[str] = None) -> str:
    """Resolve dataset name from env, file, or default."""
    if dataset_name is not None:
        return dataset_name.strip().lower()

    env_value = os.environ.get("AUTORESEARCH_DATASET")
    if env_value and env_value.strip().lower() in ("tinystories",):
        return env_value.strip().lower()

    # Check active_dataset.txt
    cache_dir = _default_cache_dir()
    active_dataset_path = os.path.join(cache_dir, "active_dataset.txt")
    if os.path.exists(active_dataset_path):
        with open(active_dataset_path, "r", encoding="utf-8") as f:
            value = f.read().strip().lower()
        if value in ("tinystories",):
            return value

    return "tinystories"


def _tokenizer_dir(dataset_name: Optional[str] = None) -> str:
    """Get tokenizer directory for a dataset."""
    dataset = _resolve_dataset_name(dataset_name)
    cache_dir = _default_cache_dir()
    return os.path.join(cache_dir, "datasets", dataset, "tokenizer")


def load_tokenizer(dataset_name: Optional[str] = None):
    """Load the BPE tokenizer from the cache directory.

    Args:
        dataset_name: Optional dataset name (resolves to tinystories by default)

    Returns:
        Tokenizer object with encode() and decode() methods

    Raises:
        FileNotFoundError: If tokenizer.pkl doesn't exist
    """
    dataset = _resolve_dataset_name(dataset_name)
    tokenizer_path = os.path.join(_tokenizer_dir(dataset), "tokenizer.pkl")

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            f"Run 'python prepare.py' to train the tokenizer first."
        )

    with open(tokenizer_path, "rb") as f:
        enc = pickle.load(f)

    # Simple wrapper that matches prepare.py.Tokenizer interface
    class SimpleTokenizer:
        def __init__(self, enc, dataset):
            self.enc = enc
            self.dataset = dataset
            self.bos_token_id = enc.encode_single_token("<|reserved_0|>")

        def encode(self, text, prepend=None):
            if prepend is not None:
                prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
            if isinstance(text, str):
                ids = self.enc.encode_ordinary(text)
                if prepend is not None:
                    ids.insert(0, prepend_id)
            else:
                raise ValueError(f"Invalid input type: {type(text)}")
            return ids

        def decode(self, ids):
            return self.enc.decode(ids)

        def decode_tokens(self, tokens):
            """Decode list of token IDs, each to its string representation."""
            return [self.enc.decode([t]) for t in tokens]

    return SimpleTokenizer(enc, dataset)


# ==========================================
# Pydantic Models for API
# ==========================================


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    max_tokens: int = Field(default=100, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(default=None, ge=1, description="Top-k sampling (optional)")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling (optional)")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    # Text-based input - easier to use
    prompt: Optional[str] = Field(default=None, description="Text prompt to continue from")
    # Token-based input - low-level control
    prompt_tokens: Optional[list[int]] = Field(
        default=None,
        description="Optional prompt tokens as list of ints. Default: BOS token"
    )


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str = Field(description="Generated text (decoded)")
    generated_tokens: list[int] = Field(description="Generated token IDs")
    num_tokens: int = Field(description="Number of tokens generated")
    prompt_used: Optional[str] = Field(default=None, description="Prompt text that was used (if text input)")
    config_used: Optional[dict] = Field(default=None, description="Model config used")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(description="Server status")
    model_loaded: bool = Field(description="Whether model is loaded")
    model_ready: bool = Field(description="Whether model is ready for inference")
    tokenizer_loaded: bool = Field(description="Whether tokenizer is loaded")
    device: str = Field(description="Device running inference (cpu/cuda)")
    checkpoint_path: Optional[str] = Field(default=None, description="Path to loaded checkpoint")
    model_cfg: Optional[dict] = Field(default=None, description="Model configuration")
    model_metrics: Optional[dict] = Field(default=None, description="Training metrics if available")
    error: Optional[str] = Field(default=None, description="Error message if model failed to load")


class ReloadModelRequest(BaseModel):
    """Request model for reloading model."""
    checkpoint: str = Field(description="Path to checkpoint file to load")


class ReloadModelResponse(BaseModel):
    """Response model for model reload."""
    status: str = Field(description="Reload status")
    checkpoint: str = Field(description="Checkpoint path used")
    device: str = Field(description="Device model is loaded on")
    config: Optional[dict] = Field(default=None, description="Model config")
    metrics: Optional[dict] = Field(default=None, description="Training metrics")
    message: str = Field(description="Detailed message")


# ==========================================
# Global State
# ==========================================

# Model and configuration
model = None
config_dict = None
metrics_dict = None

# Tokenizer
tokenizer = None
dataset_name = None

# Inference settings
device = os.environ.get("DEVICE", "cpu").lower()
dataset = os.environ.get("AUTORESEARCH_DATASET", None)

# Server state
model_ready = False
tokenizer_ready = False
model_error = None
tokenizer_error = None
checkpoint_path_used = None


# ==========================================
# Model Loading
# ==========================================


def load_model_from_checkpoint(checkpoint_path: str, target_device: Optional[str] = None) -> bool:
    """Load model from checkpoint path.

    Args:
        checkpoint_path: Path to checkpoint file
        target_device: Device to load model on (defaults to global device setting)

    Returns:
        True if successful, raises exception otherwise

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If loading fails
    """
    global model, config_dict, metrics_dict, model_ready, model_error, checkpoint_path_used

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if target_device is None:
        target_device = device

    # Force CPU inference as specified
    if target_device != "cpu":
        print(f"Warning: DEVICE={target_device} but forcing CPU inference")
        target_device = "cpu"

    try:
        print(f"Loading checkpoint from {checkpoint_path}...")
        model, config_dict, metrics_dict = load_checkpoint(checkpoint_path, device=target_device)
        print(f"Checkpoint loaded. Config: {config_dict}")
        if metrics_dict:
            print(f"Metrics: {metrics_dict}")

        model_ready = True
        model_error = None
        checkpoint_path_used = checkpoint_path
        print(f"Model loaded successfully on {target_device}")
        return True

    except Exception as e:
        model_ready = False
        model_error = str(e)
        print(f"Failed to load checkpoint: {e}")
        raise RuntimeError(f"Failed to load checkpoint: {e}")


def load_tokenizer_from_cache(dataset_name_arg: Optional[str] = None) -> bool:
    """Load the tokenizer from the cache directory.

    Args:
        dataset_name_arg: Optional dataset name to load tokenizer for

    Returns:
        True if successful, raises exception otherwise
    """
    global tokenizer, dataset_name, tokenizer_ready, tokenizer_error

    try:
        print(f"Loading tokenizer for dataset: {dataset_name_arg or 'auto-resolving'}...")
        tokenizer = load_tokenizer(dataset_name_arg)
        dataset_name = _resolve_dataset_name(dataset_name_arg)
        tokenizer_ready = True
        tokenizer_error = None
        print(f"Tokenizer loaded successfully for dataset: {dataset_name}")
        return True

    except Exception as e:
        tokenizer_ready = False
        tokenizer_error = str(e)
        print(f"Failed to load tokenizer: {e}")
        raise RuntimeError(f"Failed to load tokenizer: {e}")


# ==========================================
# FastAPI App Setup
# ==========================================

app = FastAPI(
    title="AutoResearch SDPA Inference API",
    description="""
    CPU-only inference API for AutoResearch SDPA models trained on GPU.

    Features:
    - 100% compatible with GPU checkpoints from train.py
    - Text-based generation (returns decoded text, not just tokens!)
    - Text prompts or token-based prompts
    - Sliding window attention
    - Value embeddings with gating
    - Rotary positional encoding
    - RMS normalization
    - Softcapped logits
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Lifecycle Events
# ==========================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load model and tokenizer on startup."""
    print("=" * 70)
    print("AutoResearch SDPA Inference API Starting...")
    print(f"Device: {device}")
    print(f"Dataset: {dataset or 'auto-resolve'}")
    print(f"Default checkpoint: {DEFAULT_CHECKPOINT}")
    print("=" * 70)

    # Load tokenizer
    try:
        load_tokenizer_from_cache(dataset)
    except Exception as e:
        tokenizer_error = str(e)
        print(f"Warning: Failed to load tokenizer: {e}")

    # Load default checkpoint if it exists
    try:
        if os.path.exists(DEFAULT_CHECKPOINT):
            load_model_from_checkpoint(DEFAULT_CHECKPOINT)
        else:
            print(f"Warning: Default checkpoint not found at {DEFAULT_CHECKPOINT}")
            model_error = f"Checkpoint not found: {DEFAULT_CHECKPOINT}"
    except Exception as e:
        print(f"Failed to load default checkpoint: {e}")
        model_error = str(e)

    print("-" * 70)
    yield

    # Cleanup on shutdown
    print("\nShutdown: freeing model resources...")
    global model
    if model is not None:
        del model
        model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Shutdown complete")


app.router.lifespan_context = lifespan


# ==========================================
# Endpoints
# ==========================================


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AutoResearch SDPA Inference API",
        "version": "1.0.0",
        "status": "running",
        "device": device,
        "dataset": dataset_name,
        "default_checkpoint": DEFAULT_CHECKPOINT,
        "features": {
            "text_generation": True,
            "text_prompting": True,
            "token_prompting": True,
        },
        "endpoints": {
            "health": "/health",
            "generate": "/generate (POST) - Generate text from prompt or tokens",
            "reload_model": "/reload_model (POST)",
            "docs": "/docs - Swagger UI documentation",
            "redoc": "/redoc - ReDoc documentation",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint.

    Returns server, model, and tokenizer status including configuration and metrics.
    """
    # Convert config_dict to JSON-serializable format (handle torch.dtype)
    model_cfg_serializable = None
    if model is not None and config_dict:
        model_cfg_serializable = {}
        for k, v in config_dict.items():
            if k == "compute_dtype" and hasattr(v, "__str__"):
                # Convert torch.dtype to string
                model_cfg_serializable[k] = str(v).replace("torch.", "")
            else:
                model_cfg_serializable[k] = v

    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_ready=model_ready,
        tokenizer_loaded=tokenizer_ready,
        device=device,
        checkpoint_path=checkpoint_path_used,
        model_cfg=model_cfg_serializable,
        model_metrics=metrics_dict,
        error=model_error,
    )


@app.post("/reload_model", response_model=ReloadModelResponse)
async def reload_model(request: ReloadModelRequest):
    """Reload model from a new checkpoint.

    This endpoint allows loading a different checkpoint without restarting the server.
    """
    try:
        load_model_from_checkpoint(request.checkpoint)

        # Convert config_dict to JSON-serializable format (handle torch.dtype and other non-serializable types)
        def serialize_value(v):
            """Convert a value to JSON-serializable format."""
            if hasattr(v, '__class__') and 'torch' in str(v.__class__):
                # Handle torch types
                return str(v).replace("torch.", "")
            elif isinstance(v, dict):
                return {k: serialize_value(val) for k, val in v.items()}
            elif isinstance(v, (list, tuple)):
                return [serialize_value(i) for i in v]
            else:
                return v

        config_serializable = {}
        if config_dict:
            for k, v in config_dict.items():
                config_serializable[k] = serialize_value(v)

        # Also serialize metrics_dict
        metrics_serializable = {}
        if metrics_dict:
            for k, v in metrics_dict.items():
                metrics_serializable[k] = serialize_value(v)

        return ReloadModelResponse(
            status="loaded",
            checkpoint=request.checkpoint,
            device=device,
            config=config_serializable,
            metrics=metrics_serializable,
            message=f"Model reloaded successfully from {request.checkpoint}",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text from the loaded model.

    The model generates text autoregressively based on the model's learned distribution.

    **Text-based prompting** (recommended for most use cases):
    Provide a `prompt` string to continue from that text.

    **Token-based prompting** (advanced):
    Provide `prompt_tokens` as a list of token IDs. Omit or use `null` BOS token.

    Args:
        request: Generation parameters including max_tokens, temperature, sampling params, and prompt

    Returns:
        Generated text, token IDs, and metadata

    Raises:
        HTTPException: If model/tokenizer is not loaded or generation fails
    """
    if model is None or not model_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not ready for inference. Error: {model_error}"
        )

    if tokenizer is None or not tokenizer_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Tokenizer not ready. Error: {tokenizer_error}. "
                    f"Run 'python prepare.py' to train the tokenizer."
        )

    # Set random seed if provided for reproducibility
    if request.seed is not None:
        torch.manual_seed(request.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.seed)

    # Handle prompt input
    prompt_text_used = None
    if request.prompt is not None and request.prompt.strip():
        # Use text prompt
        prompt_text_used = request.prompt
        encoded_tokens = tokenizer.encode(prompt_text_used, prepend="<|reserved_0|>")
        idx = torch.tensor([encoded_tokens], dtype=torch.long).to(device)
    elif request.prompt_tokens is not None:
        # Use token-based prompt
        idx = torch.tensor([request.prompt_tokens], dtype=torch.long).to(device)
        if prompt_text_used is None:
            prompt_text_used = tokenizer.decode(request.prompt_tokens)
    else:
        # Default: BOS token only
        idx = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)

    try:
        # Generate tokens
        generated = generate(
            model=model,
            idx=idx,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            show_progress=False,
        )

        generated_tokens = generated.tolist()[0]

        # Decode to text
        generated_text = tokenizer.decode(generated_tokens)

        # Convert config_dict to JSON-serializable format (handle torch.dtype)
        config_used_serializable = {}
        if config_dict:
            for k, v in config_dict.items():
                if k == "compute_dtype" and hasattr(v, "__str__"):
                    config_used_serializable[k] = str(v).replace("torch.", "")
                else:
                    config_used_serializable[k] = v

        return GenerateResponse(
            generated_text=generated_text,
            generated_tokens=generated_tokens,
            num_tokens=len(generated_tokens),
            prompt_used=prompt_text_used,
            config_used=config_used_serializable,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# ==========================================
# Main Entry Point
# ==========================================


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server using uvicorn."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="AutoResearch SDPA Inference API Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--checkpoint', default=None, help='Override default checkpoint path')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device to use (cpu/cuda)')
    parser.add_argument('--dataset', default=None, help='Dataset name for tokenizer (tinystories)')

    args = parser.parse_args()

    if args.checkpoint:
        os.environ['DEFAULT_CHECKPOINT'] = args.checkpoint
    if args.device:
        device = args.device
    if args.dataset:
        dataset = args.dataset

    run_server(host=args.host, port=args.port)