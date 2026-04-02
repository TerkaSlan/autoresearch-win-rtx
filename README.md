# AutoResearch Inference

> Fork of [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) with production-ready inference API and web frontend.

This fork adds a **production-ready inference API** with 100% architectural compatibility with GPU-trained checkpoints, plus a Next.js web frontend for experiment visualization.

## What Was Added

Based on the upstream [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx):

| Feature | Upstream | This Fork |
|---------|----------|-----------|
| Training (GPU) | ✅ | ✅ (same) |
| Checkpoint save with provenance | ✅ | ✅ (same) |
| **Inference module** | ❌ | **✅** EXACT model from train.py |
| **REST API server** | ❌ | **✅** FastAPI serving experiment samples |
| **Web frontend** | ❌ | **✅** Next.js dashboard |
| **Metrics computation** | ❌ | **✅** Repetition, uniqueness, constraint scoring |
| **CPU inference** | ❌ | **✅** GPU checkpoints on CPU |
| **Built-in tokenizer** | ❌ | **✅** BPE in Docker image |
| **Dockerfile for API** | ❌ | **✅** `Dockerfile-api` |
| **K8s deployment ready** | ❌ | **✅** UID 1000 non-root |

## Quick Start

### Docker (Recommended for Production)

```bash
# Build the inference API image (includes tokenizer)
docker build -t autoresearch-inference:latest -f Dockerfile-api .

# Run locally (CPU-only)
docker run -p 8000:8000 \
  -v /path/to/checkpoints:/app/checkpoints \
  -e AUTORESEARCH_CHECKPOINTS_DIR=/app/checkpoints \
  autoresearch-inference:latest

# Or push to your registry
docker tag autoresearch-inference:latest cerit.io/tslaninakova/autoresearch-inference:latest
docker push cerit.io/tslaninakova/autoresearch-inference:latest
```

### Frontend (Next.js Dashboard)

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

### Local Development (Training + Inference)

```bash
# Install dependencies
uv sync

# Prepare data and train tokenizer (creates ~/.cache/autoresearch)
uv run prepare.py

# Run training on GPU
uv run train.py

# Run inference API locally
uv run api_server.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and available endpoints |
| `/health` | GET | Health check, experiments count |
| `/docs` | GET | Interactive Swagger UI documentation |
| `/experiments` | GET | List all experiment directories |
| `/latest` | GET | Get latest experiment sample |
| `/info/{exp_name}` | GET | Detailed info about specific experiment |
| `/sample/{exp_name}` | GET | Get sample output from experiment |
| `/file/{exp_name}/{filename}` | GET | Get any file from experiment directory |

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "checkpoints_path": "/app/checkpoints",
  "experiments_found": 5
}
```

### List Experiments

```bash
curl http://localhost:8000/experiments
```

Response:
```json
[
  {
    "name": "20240317_123456-abc123",
    "path": "/app/checkpoints/20240317_123456-abc123",
    "checkpoint_exists": true,
    "sample_output": "Once upon a time...",
    "model_info": "val_bpb: 2.123",
    "git_log": "commit abc123..."
  }
]
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `AUTORESEARCH_CHECKPOINTS_DIR` | Path to checkpoints folder | `./checkpoints` |

## Project Structure

```
.
├── api_server.py          # FastAPI server (serves pre-computed samples)
├── inference.py           # Inference module (exact model from train.py)
├── train.py               # Training script with checkpoint management
├── prepare.py             # Data preparation + tokenizer training
├── program.md             # Agent instructions for experiments
├── Dockerfile             # Training environment (GPU)
├── Dockerfile-api         # Inference API environment (CPU)
├── pyproject.toml         # Main dependencies
├── pyproject-api.toml     # API-specific dependencies
├── checkpoints/           # Experiment outputs
│   ├── YYYYMMDD_HHMMSS-commit/
│   │   ├── checkpoint.pt
│   │   ├── sample_output.txt
│   │   ├── model_info.txt
│   │   └── git_log.txt
│   └── ...
└── frontend/              # Next.js web dashboard
    ├── app/
    │   ├── page.tsx       # Main dashboard (experiment browser)
    │   ├── about/page.tsx # About/documentation page
    │   └── api/           # API proxy routes
    ├── lib/
    │   └── metrics.ts     # Metrics computation utilities
    └── package.json
```

## How It Works

### Architecture Flow

1. **Training**: `train.py` runs experiments on GPU, saves checkpoints with provenance
2. **Inference**: After successful experiment, agent runs inference and saves `sample_output.txt`
3. **API**: `api_server.py` serves pre-computed samples (no dynamic model loading)
4. **Frontend**: Next.js dashboard browses experiments, visualizes metrics

### Why Pre-computed Samples?

The API serves **pre-computed samples** instead of running inference on-demand because:

- Eliminates architecture compatibility issues between training/inference environments
- No model loading needed (fast responses, just reading text files)
- Works with any model architecture the training agent produces
- Training and inference run in the same environment (guaranteed compatibility)

## Model Architecture (Ground Truth)

The inference module uses the **exact same architecture** as `train.py` for 100% compatibility:

| Component | Implementation |
|-----------|----------------|
| **Normalization** | `F.rms_norm` (Root Mean Square) |
| **Attention Backend** | `F.scaled_dot_product_attention` (SDPA) |
| **MLP Activation** | `F.relu(x).square()` (ReLU²) |
| **Rotary Encoding** | Precomputed cos/sin buffers |
| **Positional Encoding** | None (RoPE only) |
| **Value Embeddings** | Alternating layers with sigmoid gating |
| **Residual Scaling** | `resid_lambdas` & `x0_lambdas` parameters |
| **Output Layer** | Softcapped logits (15 * tanh(x/15)) |
| **Sliding Window** | Dynamic window sizes per layer |
| **GQA** | `enable_gqa` parameter for grouped-query attention |

## Checkpoint Format

### Wrapped Format (recommended)
```python
{
  'model_state_dict': {...},
  'config': {...},
  'metrics': {'val_bpb': ...},
  'timestamp': '...',
  'is_best': True
}
```

### Plain State Dict Format
```python
{
  'transformer.wte.weight': tensor(...),
  'transformer.h.0.attn.c_q.weight': tensor(...),
  ...
}
```

When loading plain checkpoints, configuration is automatically derived from tensor shapes.

## Experiment Tracking Features

### Checkpoint History
Each training run saves timestamped checkpoints (`checkpoint_YYYYMMDD_HHMMSS_pre_eval.pt`) before evaluation, enabling full experiment history recovery.

### Best Model Tracking
Automatically maintains `checkpoint_best.pt` with the lowest validation bits-per-byte across all runs.

### Experimental Provenance
Successfully marked experiments save a complete provenance package:
- `run_info.json` - full metadata including config, runtime info, metrics
- `program.md` - copy of agent instructions used
- `results.tsv` - copy of results file for analysis

## Docker Images

### `Dockerfile` (Training - Upstream)
- Base: `nvidia/cu12.6.0-devel-ubuntu22.04`
- PyTorch: 2.9.1+cu128 (GPU)
- User: UID 1000
- Purpose: Training on GPU (RTX PRO 6000 Blackwell compatible)

### `Dockerfile-api` (Inference - Added by this fork)
- Base: `python:3.10-slim`
- PyTorch: CPU-only
- User: UID 1000 (for K8s securityContext)
- Includes: BPE tokenizer (trained on TinyStories)
- Purpose: Production inference API on CPU

## GPU Support (Training Only)

Supported GPUs:

| Architecture | Compute Capability | Minimum VRAM | Example Models |
|--------------|-------------------|-------------|----------------|
| Blackwell | sm_100 | 48GB | RTX PRO 6000 Blackwell |
| Ada | sm_89 | 10GB | RTX 4090, RTX 4080 |
| Ampere | sm_86 | 10GB | RTX 3090, RTX 3080 |
| Turing | sm_75 | 8GB | RTX 2080 Ti |

**Inference runs on CPU** - GPU checkpoints are transparently loaded on CPU with proper dtype conversion.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- **Training**: NVIDIA GPU (Turing >=8GB or Ampere/Ada/Blackwell >=10GB)
- **Inference**: CPU-only (GPU not required)
- Node.js 18+ (for frontend)
- Docker optional (required for K8s deployment)

## License

MIT

## Acknowledgments

- Original: [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- Windows/RTX fork: [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx)
