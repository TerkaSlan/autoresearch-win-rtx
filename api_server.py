#!/usr/bin/env python3
"""
FastAPI server for AutoResearch SDPA - serves pre-computed experiment samples.

Instead of loading models dynamically, this server reads pre-generated samples
from experiment directories. The training agent runs inference after each successful
experiment and saves the sample to the experiment directory.

This eliminates architecture compatibility issues since inference runs in the
same environment as training.
"""

import os
import json
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

# Checkpoint directories from environment
CHECKPOINTS_BASE_PATH = os.environ.get(
    "AUTORESEARCH_CHECKPOINTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
)

# ==========================================
# Pydantic Models
# ==========================================

class ExperimentInfo(BaseModel):
    """Information about an experiment."""
    name: str
    path: str
    checkpoint_exists: bool
    sample_output: Optional[str] = None
    model_info: Optional[str] = None
    git_log: Optional[str] = None


class SampleResponse(BaseModel):
    """Response containing a pre-computed sample."""
    experiment: str
    sample_path: str
    sample_content: Optional[str] = None
    model_info: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    checkpoints_path: str
    experiments_found: int


class APIRootResponse(BaseModel):
    """Root endpoint response."""
    name: str
    version: str
    description: str
    checkpoints_path: str
    endpoints: dict


# ==========================================
# Helper Functions
# ==========================================

def find_experiment_directories() -> List[Path]:
    """Find all experiment directories in the checkpoints folder."""
    experiments = []
    checkpoints_path = Path(CHECKPOINTS_BASE_PATH)

    if not checkpoints_path.exists():
        return experiments

    # Look for directories matching the pattern: timestamp-commit_hash
    for item in checkpoints_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it looks like an experiment directory
            if (item / 'checkpoint.pt').exists() or (item / 'sample_output.txt').exists():
                experiments.append(item)

    # Sort by name (timestamp-based) to get newest first
    experiments.sort(key=lambda x: x.name, reverse=True)
    return experiments


def get_latest_experiment() -> Optional[Path]:
    """Get the latest experiment directory."""
    experiments = find_experiment_directories()
    return experiments[0] if experiments else None


def read_experiment_file(exp_dir: Path, filename: str) -> Optional[str]:
    """Read a file from an experiment directory."""
    file_path = exp_dir / filename
    if file_path.exists():
        try:
            return file_path.read_text()
        except Exception:
            return None
    return None


# ==========================================
# FastAPI App
# ==========================================

try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Install with: uv pip install fastapi uvicorn")

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="AutoResearch SDPA - Experiment Samples API",
        description="""
API server that serves pre-computed samples from AutoResearch experiments.

## How it works

1. Training agent runs experiments and saves checkpoints
2. After each successful experiment, the agent runs inference and saves sample_output.txt
3. This API serves those pre-computed samples

## Benefits

- No model loading needed (no architecture compatibility issues)
- Fast responses (just reading text files)
- Works with any model architecture the training agent produces
        """,
        version="2.0.0"
    )

    # Enable CORS for all origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ==========================================
    # Endpoints
    # ==========================================

    @app.get("/", response_model=APIRootResponse)
    async def root():
        """Root endpoint with API information."""
        return APIRootResponse(
            name="AutoResearch SDPA - Experiment Samples API",
            version="2.0.0",
            description="Serves pre-computed samples from AutoResearch experiments",
            checkpoints_path=str(CHECKPOINTS_BASE_PATH),
            endpoints={
                "health": "/health",
                "list_experiments": "/experiments",
                "latest_sample": "/latest",
                "experiment_info": "/info/{exp_name}",
                "sample_file": "/sample/{exp_name}"
            }
        )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        experiments = find_experiment_directories()
        return HealthResponse(
            status="ok",
            checkpoints_path=str(CHECKPOINTS_BASE_PATH),
            experiments_found=len(experiments)
        )

    @app.get("/experiments", response_model=List[ExperimentInfo])
    async def list_experiments():
        """List all available experiments."""
        experiments = find_experiment_directories()
        result = []

        for exp_dir in experiments:
            info = ExperimentInfo(
                name=exp_dir.name,
                path=str(exp_dir),
                checkpoint_exists=(exp_dir / 'checkpoint.pt').exists(),
                sample_output=read_experiment_file(exp_dir, 'sample_output.txt'),
                model_info=read_experiment_file(exp_dir, 'model_info.txt'),
                git_log=read_experiment_file(exp_dir, 'git_log.txt')
            )
            result.append(info)

        return result

    @app.get("/latest", response_model=SampleResponse)
    async def latest_sample():
        """Get the sample from the latest experiment."""
        latest = get_latest_experiment()

        if latest is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No experiments found"
            )

        sample_content = read_experiment_file(latest, 'sample_output.txt')
        model_info = read_experiment_file(latest, 'model_info.txt')

        return SampleResponse(
            experiment=latest.name,
            sample_path=str(latest / 'sample_output.txt'),
            sample_content=sample_content,
            model_info=model_info
        )

    @app.get("/info/{exp_name}", response_model=ExperimentInfo)
    async def experiment_info(exp_name: str):
        """Get detailed information about a specific experiment."""
        exp_path = Path(CHECKPOINTS_BASE_PATH) / exp_name

        if not exp_path.exists() or not exp_path.is_dir():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment not found: {exp_name}"
            )

        return ExperimentInfo(
            name=exp_path.name,
            path=str(exp_path),
            checkpoint_exists=(exp_path / 'checkpoint.pt').exists(),
            sample_output=read_experiment_file(exp_path, 'sample_output.txt'),
            model_info=read_experiment_file(exp_path, 'model_info.txt'),
            git_log=read_experiment_file(exp_path, 'git_log.txt')
        )

    @app.get("/sample/{exp_name}", response_model=SampleResponse)
    async def get_sample(exp_name: str):
        """Get the sample output from a specific experiment."""
        exp_path = Path(CHECKPOINTS_BASE_PATH) / exp_name

        if not exp_path.exists() or not exp_path.is_dir():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment not found: {exp_name}"
            )

        sample_content = read_experiment_file(exp_path, 'sample_output.txt')
        model_info = read_experiment_file(exp_path, 'model_info.txt')

        if sample_content is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No sample output found for experiment: {exp_name}"
            )

        return SampleResponse(
            experiment=exp_path.name,
            sample_path=str(exp_path / 'sample_output.txt'),
            sample_content=sample_content,
            model_info=model_info
        )

else:
    # Create a minimal app for testing when FastAPI is not available
    print("WARNING: FastAPI not available. API server will not function.")


if __name__ == '__main__':
    if FASTAPI_AVAILABLE:
        print("="*60)
        print("AutoResearch SDPA Samples API Server Starting...")
        print(f"Checkpoints path: {CHECKPOINTS_BASE_PATH}")
        print(f"Experiments found: {len(find_experiment_directories())}")
        print("="*60)
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("Cannot start server: FastAPI not installed")
