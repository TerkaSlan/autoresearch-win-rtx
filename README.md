# AutoResearch Inference

> Fork of [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) with inference API support.

This is a fork of the Windows RTX version with added inference capabilities:
- **REST API** for text generation
- **Automatic model loading** on startup
- **Checkpoint format compatibility** - loads both wrapped and original repo formats
- **GPU/CPU agnostic** - automatically detects hardware
- **Docker support** - single command deployment

## Features

Based on the upstream [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and forked from [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx), this version adds:

1. **Standalone inference module** - `model.py` with complete architecture
2. **Checkpoint loading** - `inference.py` handles both state_dict formats
3. **FastAPI server** - `api_server.py` with auto-load on startup
4. **Docker image** - Production-ready container

## Quick Start

### Docker (Recommended)

```bash
# Build the image
docker build -t autoresearch-inference .

# Run the server
docker run --gpus all -p 8000:8000 \
  -v /path/to/checkpoints:/app/checkpoints \
  autoresearch-inference

# Access API
curl http://localhost:8000/health
curl http://localhost:8000/docs  # Interactive documentation
```

### Local Development

For training and experiments, follow the original README instructions:

```bash
# Install dependencies
uv sync

# Prepare data and train tokenizer
uv run prepare.py

# Run training
uv run train.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check and model status |
| `/docs` | GET | Interactive API documentation |
| `/reload_model` | POST | Reload a checkpoint |
| `/generate` | POST | Generate text |

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_ready": true,
  "device": "cuda",
  "model_config_dict": {...},
  "model_error": null
}
```

### Generate Text

```bash
POST /generate
Content-Type: application/json

{
  "max_tokens": 100,
  "temperature": 1.0,
  "top_k": 50,
  "top_p": 0.95,
  "seed": 42
}
```

Response:
```json
{
  "generated_tokens": [0, 123, 456, ...],
  "num_tokens": 100,
  "config_used": {...}
}
```

## Configuration

Set the `DEFAULT_CHECKPOINT` environment variable to auto-load a specific checkpoint:

```bash
export DEFAULT_CHECKPOINT=/path/to/checkpoint.pt
python api_server.py
```

Or with Docker:
```bash
docker run --gpus all -p 8000:8000 \
  -v /path/to/checkpoints:/app/checkpoints \
  -e DEFAULT_CHECKPOINT=/app/checkpoints/my-checkpoint.pt \
  autoresearch-inference
```

## Checkpoint Format Compatibility

The inference module handles checkpoint loading for both formats:

1. **Wrapped format** (created by `save_checkpoint` in inference.py):
   ```python
   torch.save({
       'model_state_dict': model.state_dict(),
       'config': {...},
       'metrics': {...}
   }, 'checkpoint.pt')
   ```

2. **Original repo format** (plain state_dict from train.py):
   ```python
   torch.save(model.state_dict(), 'checkpoint.pt')
   ```

When loading plain state_dict checkpoints, configuration is automatically inferred from tensor shapes.

## GPU Support

This implementation uses PyTorch SDPA attention instead of Flash Attention, providing native support for:

### Supported GPUs

| Architecture | Compute Capability | Minimum VRAM | Example Models |
|--------------|-------------------|-------------|----------------|
| Blackwell | sm_100 | 48GB | RTX PRO 6000 Blackwell |
| Ada | sm_89 | 10GB | RTX 4090, RTX 4080, RTX PRO 6000 Ada |
| Ampere | sm_86 | 10GB | RTX 3090, RTX 3080, RTX PRO 6000 Ampere |
| Turing | sm_75 | 8GB | RTX 2080 Ti, RTX PRO 4000 |

### NVIDIA RTX PRO 6000 Blackwell

The NVIDIA RTX PRO 6000 Blackwell (sm_100) is fully supported with:
- 48GB GDDR7 VRAM - ideal for larger models and batch sizes
- PyTorch 2.9.1 with CUDA 12.6 support
- Native PyTorch SDPA attention (no flash-attn required)
- BF16/TF32 precision support

**Testing with RTX PRO 6000 Blackwell:**
```bash
# Verify GPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Expected output:
# CUDA available: True
# GPU: NVIDIA RTX PRO 6000
```

## Differences from Upstream

Based on [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx):

| Feature | Upstream | This Fork |
|---------|----------|-----------|
| Training | Yes | Yes (same) |
| Inference module | No | **Yes** |
| REST API | No | **Yes** |
| Docker | No | **Yes** |
| Auto-load model | No | **Yes** |
| Checkpoint compatibility | Single format | **Both formats** |

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Optional: NVIDIA GPU (Turing >=8GB or Ampere/Ada/Blackwell >=10GB)
- Optional: Docker for containerized deployment

## License

MIT

## Acknowledgments

- Original: [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- Windows/RTX fork: [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx)