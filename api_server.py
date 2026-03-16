#!/usr/bin/env python3
"""
FastAPI server for AutoResearch SDPA model inference.

Checkpoint loads automatically on startup from DEFAULT_CHECKPOINT environment variable.
"""

import torch
import os
from inference import load_checkpoint, generate
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Checkpoint path from environment
DEFAULT_CHECKPOINT = os.environ.get("DEFAULT_CHECKPOINT", "checkpoints/checkpoint_pre_eval.pt")

# Try to import FastAPI, handle if not installed
try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Install with: uv pip install fastapi uvicorn")

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

    # Global model state
    model = None
    config = None
    metrics = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_ready = False
    model_error = None

    # ==========================================
    # Pydantic Models
    # ==========================================

    class GenerateRequest(BaseModel):
        max_tokens: int = 100
        temperature: float = 1.0
        top_k: Optional[int] = None
        top_p: Optional[float] = None
        seed: Optional[int] = None

    class GenerateResponse(BaseModel):
        generated_tokens: list
        num_tokens: int
        config_used: Optional[dict] = None

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        model_ready: bool
        device: str
        model_config_dict: Optional[dict] = None
        model_error: Optional[str] = None

    class ReloadModelRequest(BaseModel):
        checkpoint: str

    class ReloadModelResponse(BaseModel):
        status: str
        config: Optional[dict] = None
        metrics: Optional[dict] = None
        device: str
        message: str

    # ==========================================
    # Model Loading Functions
    # ==========================================

    def load_model_from_checkpoint(checkpoint_path: str):
        """Load model from checkpoint path."""
        global model, config, metrics, model_ready, model_error

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model, config, metrics = load_checkpoint(checkpoint_path)
        model.to(device)
        model_ready = True
        model_error = None
        print(f"Model loaded from {checkpoint_path}. Device: {device}")
        return True

    # ==========================================
    # Startup
    # ==========================================

    @app.on_event("startup")
    async def startup_event():
        print("="*60)
        print("AutoResearch SDPA API Server Starting...")
        print(f"Device: {device}")
        print(f"FastAPI: {FASTAPI_AVAILABLE}")
        print(f"Default checkpoint: {DEFAULT_CHECKPOINT}")
        print("="*60)

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
            "endpoints": {
                "health": "/health",
                "generate": "/generate (POST)",
                "reload_model": "/reload_model (POST)",
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
            model_error=model_error
        )

    @app.post("/reload_model", response_model=ReloadModelResponse)
    async def reload_model(request: ReloadModelRequest):
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
    async def generate_text(request: GenerateRequest):
        """Generate text from the loaded model."""
        if model is None or not model_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model not loaded or not ready. Error: {model_error}"
            )

        # Set random seed if provided
        if request.seed is not None:
            torch.manual_seed(request.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(request.seed)

        try:
            # Generate
            generated = generate(
                model,
                torch.tensor([[0]], dtype=torch.long).to(device),
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                show_progress=False
            )

            return GenerateResponse(
                generated_tokens=generated.tolist()[0],
                num_tokens=len(generated.tolist()[0]),
                config_used=config
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # For direct running
    def run_server(host: str = "0.0.0.0", port: int = 8000):
        uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    args = parser.parse_args()

    if FASTAPI_AVAILABLE:
        run_server(host=args.host, port=args.port)
    else:
        print("FastAPI not available. Install with: uv pip install fastapi uvicorn")
        exit(1)