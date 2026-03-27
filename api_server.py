#!/usr/bin/env python3
"""
FastAPI server for AutoResearch SDPA model inference.

Features:
- CPU-only inference for GPU-trained checkpoints
- /health endpoint - check server and model status
- /generate endpoint - generate text (returns both tokens and decoded text)
- /list endpoint - list all experiment directories
- /info endpoint - get info about latest experiment
- /info/{exp_dir} endpoint - get info about specific experiment
- Automatic checkpoint loading from environment or file
- File-based caching for fast repeated requests
"""

import torch
import os
import json
import pickle
import subprocess
import hashlib
from pathlib import Path
from typing import Optional
from collections import namedtuple
from contextlib import asynccontextmanager

# Import inference module
from inference import load_checkpoint, generate

# Checkpoint path from environment
DEFAULT_CHECKPOINT = os.environ.get("AUTORESEARCH_CHECKPOINT",
                                    os.environ.get("DEFAULT_CHECKPOINT", "checkpoints/checkpoint_pre_eval.pt"))
CACHE_DIR = os.environ.get("AUTORESEARCH_CACHE_DIR", "/app/checkpoints/cache")
CHECKPOINTS_BASE_PATH = os.environ.get(
    "AUTORESEARCH_CHECKPOINTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
)

# API Key authentication
API_KEY = os.environ.get("AUTORESEARCH_API_KEY")
API_KEY_PREFIX = "Bearer "

# ==========================================
# File-based Cache
# ==========================================

class FileCache:
    """Simple file-based cache that persists to disk."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cache initialized at: {self.cache_dir}")

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{key_hash}.json"

    def get(self, key: str) -> Optional[dict]:
        """Get cached response if exists."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Cache read error for {key}: {e}")
                return None
        return None

    def set(self, key: str, value: dict) -> None:
        """Store response in cache."""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'w') as f:
                json.dump(value, f)
        except IOError as e:
            print(f"Cache write error for {key}: {e}")

    def clear(self, key_prefix: Optional[str] = None) -> int:
        """Clear cache entries. If key_prefix provided, only clear matching entries."""
        cleared = 0
        for cache_file in self.cache_dir.glob("*.json"):
            if key_prefix:
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        if data.get('_cache_key', '').startswith(key_prefix):
                            cache_file.unlink()
                            cleared += 1
                except (json.JSONDecodeError, IOError):
                    pass
            else:
                cache_file.unlink()
                cleared += 1
        return cleared

    def get_stats(self) -> dict:
        """Get cache statistics."""
        files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "cache_dir": str(self.cache_dir),
            "entries": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }

# Global cache instance
cache = None

# API Key authentication dependency - defined after FastAPI imports below

# Try to import FastAPI
try:
    from fastapi import FastAPI, HTTPException, status, Depends
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Install with: uv pip install fastapi uvicorn")

# Global instances (must be after FastAPI imports)
if FASTAPI_AVAILABLE:
    security = HTTPBearer(auto_error=False)


# ==========================================
# API Key Authentication (after FastAPI imports)
# ==========================================

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    """Verify API key from Bearer token."""
    if not API_KEY:
        print("Warning: AUTORESEARCH_API_KEY not set - API key validation disabled")
        return True  # Allow if no key configured

    if credentials is None or credentials.credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Please provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return True


# ==========================================
# Helper Functions for Experiment Directories
# ==========================================

def _parse_git_log(git_log_path: Path) -> list:
    """Parse git_log.txt and return list of commits."""
    commits = []
    if not git_log_path.exists():
        return commits

    content = git_log_path.read_text()
    lines = content.split('\n')
    current_commit = None

    for line in lines:
        line = line.rstrip()
        if line.startswith('commit '):
            if current_commit:
                commits.append(current_commit)
            sha = line.split(' ')[1].strip()
            current_commit = {
                'commit_sha': sha,
                'commit_short': sha[:7] if len(sha) >= 7 else sha,
                'author': '',
                'date': '',
                'message': ''
            }
        elif line.startswith('Author: '):
            if current_commit:
                current_commit['author'] = line.replace('Author: ', '').strip()
        elif line.startswith('Date:   '):
            if current_commit:
                current_commit['date'] = line.replace('Date:   ', '').strip()
        elif current_commit and line and not line.startswith('\t'):
            if current_commit['message']:
                current_commit['message'] += '\n'
            current_commit['message'] += line
        elif line.startswith('\t') and current_commit:
            if current_commit['message']:
                current_commit['message'] += '\n'
            current_commit['message'] += line[1:]

    if current_commit:
        commits.append(current_commit)

    return commits


def _parse_results_tsv(results_path: Path) -> Optional[dict]:
    """Parse results.tsv file."""
    if not results_path.exists():
        return None

    content = results_path.read_text()
    lines = [line.rstrip() for line in content.split('\n') if line.strip()]

    if not lines:
        return {'header': [], 'rows': []}

    return {
        'header': lines[0].split('\t'),
        'rows': [row.split('\t') for row in lines[1:]]
    }


def _get_exp_dir_path(exp_dir_name: str) -> Path:
    """Get the full path to an experiment directory."""
    return Path(CHECKPOINTS_BASE_PATH) / exp_dir_name


def _get_latest_experiment_dir() -> Optional[str]:
    """Find the most recent experiment directory with a checkpoint."""
    base_path = Path(CHECKPOINTS_BASE_PATH)
    if not base_path.exists():
        return None

    exp_dirs = []
    for item in base_path.iterdir():
        if item.is_dir():
            name = item.name
            if name[:4].isdigit() and len(name) >= 8:
                checkpoint_path = item / "checkpoint.pt"
                if checkpoint_path.exists():
                    try:
                        exp_dirs.append((name, item))
                    except (ValueError, IndexError):
                        continue

    if not exp_dirs:
        return None

    exp_dirs.sort(key=lambda x: x[0], reverse=True)
    return exp_dirs[0][0]


def _list_experiment_dirs() -> list:
    """List all experiment directories."""
    base_path = Path(CHECKPOINTS_BASE_PATH)
    if not base_path.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {CHECKPOINTS_BASE_PATH}")

    experiments = []
    for item in base_path.iterdir():
        if item.is_dir():
            name = item.name
            if name[:4].isdigit() and len(name) >= 8:
                exp_dir_path = item
                checkpoint_path = exp_dir_path / "checkpoint.pt"
                has_checkpoint = checkpoint_path.exists()
                experiments.append({
                    'name': name,
                    'checkpoint_path': str(checkpoint_path) if has_checkpoint else None,
                    'has_checkpoint': has_checkpoint
                })

    experiments.sort(key=lambda x: x['name'], reverse=True)
    return experiments


def _get_exp_dir_info(exp_dir_name: str) -> dict:
    """Gather information about an experiment directory."""
    exp_dir = _get_exp_dir_path(exp_dir_name)

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    checkpoint_path = exp_dir / "checkpoint.pt"
    program_path = exp_dir / "program.md"
    results_global_path = exp_dir / "results.tsv.global"
    git_log_path = exp_dir / "git_log.txt"
    metadata_path = exp_dir / "run_metadata.json"

    has_checkpoint = checkpoint_path.exists()
    info = {
        'exp_dir': exp_dir_name,
        'checkpoint_path': str(checkpoint_path) if has_checkpoint else None,
        'has_checkpoint': has_checkpoint,
        'has_program': program_path.exists(),
        'has_results': results_global_path.exists(),
        'has_global_results': results_global_path.exists(),
        'has_git_log': git_log_path.exists(),
        'has_metadata': metadata_path.exists(),
    }

    if git_log_path.exists():
        info['git_log'] = _parse_git_log(git_log_path)

    if results_global_path.exists():
        info['results_data'] = _parse_results_tsv(results_global_path)

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            info['metadata'] = json.load(f)

    return info


# ==========================================
# Global State
# ==========================================

model = None
config = None
metrics = None
device = 'cpu'  # Force CPU inference as specified
model_ready = False
model_error = None
checkpoint_path_used = None

# Try to import tokenizer support
try:
    from prepare import Tokenizer
    TOKENIZER_AVAILABLE = True
    # Load tokenizer from checkpoints directory
    tokenizer = Tokenizer.from_directory(tokenizer_dir="/app/checkpoints/tokenizer")
    tokenizer_ready = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    tokenizer = None
    tokenizer_ready = False
    print("Tokenizer not available. Some features may be limited.")
except Exception as e:
    tokenizer = None
    tokenizer_ready = False
    print(f"Failed to load tokenizer: {e}")


# ==========================================
# Pydantic Models
# ==========================================

class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    max_tokens: int = Field(default=100, ge=1, le=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Sampling temperature (lower = more focused)")
    top_k: Optional[int] = Field(default=50, ge=1, description="Top-k sampling (50 is a good default)")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling (0.9 is a good default)")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    checkpoint: Optional[str] = Field(default=None, description="Path to checkpoint file to use for generation")


class GenerateResponse(BaseModel):
    generated_tokens: list
    generated_text: Optional[str] = None
    num_tokens: int
    prompt_used: Optional[str] = None
    config_used: Optional[dict] = None
    cached: bool = False


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_ready: bool
    device: str
    model_config_dict: Optional[dict] = None
    model_error: Optional[str] = None
    checkpoint_path: Optional[str] = None


class ReloadModelRequest(BaseModel):
    checkpoint: str


class ReloadModelResponse(BaseModel):
    status: str
    config: Optional[dict] = None
    metrics: Optional[dict] = None
    device: str
    message: str


class GitLogCommit(BaseModel):
    commit_sha: str
    commit_short: str
    author: str
    date: str
    message: str


class ExperimentInfoResponse(BaseModel):
    exp_dir: str
    checkpoint_path: Optional[str] = None
    has_checkpoint: bool
    has_program: bool
    has_results: bool
    has_git_log: bool
    has_metadata: bool
    git_log: Optional[list] = None
    results_data: Optional[dict] = None
    metadata: Optional[dict] = None


class LatestExperimentResponse(BaseModel):
    exp_dir: str
    checkpoint_path: Optional[str] = None


class ExperimentDir(BaseModel):
    name: str
    checkpoint_path: Optional[str] = None
    has_checkpoint: bool


class ExperimentListResponse(BaseModel):
    experiments: list
    count: int


# ==========================================
# Model Loading Functions
# ==========================================

def load_model_from_checkpoint(checkpoint_path: str):
    """Load model from checkpoint path."""
    global model, config, metrics, model_ready, model_error, checkpoint_path_used

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, config, metrics = load_checkpoint(checkpoint_path)
    model.to(device)
    model_ready = True
    model_error = None
    checkpoint_path_used = checkpoint_path
    print(f"Model loaded from {checkpoint_path}. Device: {device}")
    return True


# ==========================================
# FastAPI App Setup
# ==========================================

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="AutoResearch SDPA Inference API",
        description="API for text generation using AutoResearch SDPA models",
        version="1.0.0"
    )

    # Enable CORS for all origins (for development)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ==========================================
# Startup
# ==========================================

@app.on_event("startup")
async def startup_event():
    global cache

    print("="*60)
    print("AutoResearch SDPA API Server Starting...")
    print(f"Device: {device}")
    print(f"FastAPI: {FASTAPI_AVAILABLE}")
    print(f"Default checkpoint: {DEFAULT_CHECKPOINT}")
    print(f"Checkpoints base path: {CHECKPOINTS_BASE_PATH}")
    print("="*60)

    # Initialize cache
    cache = FileCache(CACHE_DIR)
    print(f"Cache initialized: {cache.get_stats()}")

    try:
        if os.path.exists(DEFAULT_CHECKPOINT):
            print(f"Loading default checkpoint from {DEFAULT_CHECKPOINT}...")
            load_model_from_checkpoint(DEFAULT_CHECKPOINT)
            print(f"Config: {config}")
            if metrics:
                print(f"Metrics: {metrics}")
        else:
            print(f"Warning: Default checkpoint not found at {DEFAULT_CHECKPOINT}")
            model_error = f"Checkpoint not found: {DEFAULT_CHECKPOINT}"
            model_ready = False
    except Exception as e:
        print(f"Failed to load model on startup: {e}")
        model_error = str(e)
        model_ready = False


# ==========================================
# Endpoints
# ==========================================

@app.get("/", response_model=dict)
async def root():
    return {
        "name": "AutoResearch SDPA Inference API",
        "version": "1.0.0",
        "status": "running",
        "default_checkpoint": DEFAULT_CHECKPOINT,
        "checkpoints_base_path": CHECKPOINTS_BASE_PATH,
        "endpoints": {
            "health": "/health",
            "generate": "/generate (POST)",
            "reload_model": "/reload_model (POST)",
            "list": "/list",
            "latest": "/latest",
            "info": "/info",
            "info_exp_dir": "/info/{exp_dir}",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_ready=model_ready,
        device=device,
        model_config_dict=config if model else None,
        model_error=model_error,
        checkpoint_path=checkpoint_path_used
    )


@app.post("/reload_model", response_model=ReloadModelResponse)
async def reload_model(
    request: ReloadModelRequest,
    auth: bool = Depends(verify_api_key)
):
    """Reload model from a new checkpoint."""
    try:
        load_model_from_checkpoint(request.checkpoint)
        return ReloadModelResponse(
            status="loaded",
            config=config,
            metrics=metrics,
            device=device,
            message=f"Model reloaded from {request.checkpoint}"
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    auth: bool = Depends(verify_api_key)
):
    """Generate text from the loaded model. Results are cached."""
    if model is None or not model_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded or not ready. Error: {model_error}"
        )

    # Determine which checkpoint to use
    target_checkpoint = request.checkpoint if request.checkpoint else checkpoint_path_used

    # Build cache key from request parameters and checkpoint
    cache_key_parts = [
        f"checkpoint:{target_checkpoint}",
        f"prompt:{request.prompt or ''}",
        f"max_tokens:{request.max_tokens}",
        f"temperature:{request.temperature}",
        f"top_k:{request.top_k}",
        f"top_p:{request.top_p}",
        f"seed:{request.seed}",
    ]
    cache_key = "|".join(cache_key_parts)

    # Check cache first
    cached_response = cache.get(cache_key)
    if cached_response:
        cached_response['cached'] = True
        return GenerateResponse(**cached_response)

    # Set random seed if provided
    if request.seed is not None:
        torch.manual_seed(request.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.seed)

    try:
        # Prepare input tokens
        # Get BOS token ID from tokenizer
        bos_token_id = None
        if hasattr(model, 'get_bos_token_id'):
            bos_token_id = model.get_bos_token_id()
        elif hasattr(model, 'enc') and hasattr(model.enc, 'encode_single_token'):
            try:
                bos_token_id = model.enc.encode_single_token("<|reserved_0|>")
            except:
                bos_token_id = None

        if request.prompt and hasattr(model, 'encode'):
            # Encode prompt with BOS prepended (required for model)
            prompt_tokens = model.encode(request.prompt, prepend=bos_token_id)
            input_tokens = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
            num_prompt_tokens = len(prompt_tokens)
        else:
            # Default: BOS token only
            if bos_token_id is not None:
                input_tokens = torch.tensor([[bos_token_id]], dtype=torch.long).to(device)
            else:
                input_tokens = torch.tensor([[0]], dtype=torch.long).to(device)
            num_prompt_tokens = 1

        # Generate
        generated = generate(
            model,
            input_tokens,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            show_progress=False
        )

        # Extract only the new tokens (continuation after prompt)
        all_tokens = generated[0].tolist()
        new_tokens = all_tokens[num_prompt_tokens:]

        # Decode tokens to text using the tokenizer
        generated_text = None
        if tokenizer is not None and hasattr(tokenizer, 'decode'):
            generated_text = tokenizer.decode(new_tokens)

        # Serialize config for JSON storage (convert torch types to strings)
        def serialize_config(cfg):
            """Convert config to JSON-serializable format."""
            if cfg is None:
                return None
            result = {}
            for k, v in cfg.items():
                if hasattr(v, '__str__') and 'torch' in str(v.__class__):
                    result[k] = str(v).replace('torch.', '')
                else:
                    result[k] = v
            return result

        response_data = {
            'generated_tokens': new_tokens,
            'generated_text': generated_text,
            'num_tokens': len(new_tokens),
            'prompt_used': request.prompt,
            'config_used': serialize_config(config),
            '_cache_key': cache_key
        }

        # Cache the response
        cache.set(cache_key, response_data)
        response_data['cached'] = False

        return GenerateResponse(**response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/list", response_model=ExperimentListResponse)
async def list_experiments(
    auth: bool = Depends(verify_api_key)
):
    """List all experiment directories."""
    try:
        experiments = _list_experiment_dirs()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return ExperimentListResponse(
        experiments=[ExperimentDir(**exp) for exp in experiments],
        count=len(experiments)
    )


@app.get("/latest", response_model=LatestExperimentResponse)
async def get_latest_exp(
    auth: bool = Depends(verify_api_key)
):
    """Get the latest experiment directory with a checkpoint."""
    latest = _get_latest_experiment_dir()
    if latest is None:
        raise HTTPException(
            status_code=404,
            detail=f"No experiment directories with checkpoint found in {CHECKPOINTS_BASE_PATH}"
        )

    exp_dir = _get_exp_dir_path(latest)
    checkpoint_path = exp_dir / "checkpoint.pt"

    return LatestExperimentResponse(
        exp_dir=latest,
        checkpoint_path=str(checkpoint_path) if checkpoint_path.exists() else None
    )


@app.get("/info/{exp_dir}", response_model=ExperimentInfoResponse)
async def get_exp_info(
    exp_dir: str,
    auth: bool = Depends(verify_api_key)
):
    """Get information about a specific experiment directory."""
    # Check cache first
    cache_key = f"info:{exp_dir}"
    if cache is not None:
        cached_response = cache.get(cache_key)
        if cached_response:
            return ExperimentInfoResponse(**cached_response)

    try:
        info = _get_exp_dir_info(exp_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    response_data = {
        'exp_dir': info['exp_dir'],
        'checkpoint_path': info['checkpoint_path'],
        'has_checkpoint': info['has_checkpoint'],
        'has_program': info['has_program'],
        'has_results': info['has_results'],
        'has_git_log': info['has_git_log'],
        'has_metadata': info['has_metadata'],
        'git_log': info.get('git_log'),
        'results_data': info.get('results_data'),
        'metadata': info.get('metadata')
    }

    # Cache the response
    if cache is not None:
        cache.set(cache_key, response_data)

    return ExperimentInfoResponse(**response_data)


@app.get("/info", response_model=ExperimentInfoResponse)
async def get_latest_exp_info(
    auth: bool = Depends(verify_api_key)
):
    """Get information about the latest experiment directory with a checkpoint."""
    latest = _get_latest_experiment_dir()
    if latest is None:
        raise HTTPException(
            status_code=404,
            detail=f"No experiment directories with checkpoint found in {CHECKPOINTS_BASE_PATH}"
        )

    try:
        info = _get_exp_dir_info(latest)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return ExperimentInfoResponse(
        exp_dir=info['exp_dir'],
        checkpoint_path=info['checkpoint_path'],
        has_checkpoint=info['has_checkpoint'],
        has_program=info['has_program'],
        has_results=info['has_results'],
        has_git_log=info['has_git_log'],
        has_metadata=info['has_metadata'],
        git_log=info.get('git_log'),
        results_data=info.get('results_data'),
        metadata=info.get('metadata')
    )


# For direct running
def run_server(host: str = "0.0.0.0", port: int = 8000):
    if FASTAPI_AVAILABLE:
        import uvicorn
        uvicorn.run(app, host=host, port=port)
    else:
        print("FastAPI not available. Install with: uv pip install fastapi uvicorn")
        exit(1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)
