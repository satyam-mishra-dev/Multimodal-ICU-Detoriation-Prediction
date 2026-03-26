import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException

from ..schemas import ModelCheckpoint, ModelLoadResponse
from ..services import inference_service as inf_svc

router = APIRouter(prefix="/models", tags=["Models"])

_SRC = Path(__file__).parent.parent.parent / "src"
OUTPUTS_DIR = _SRC / "outputs"


def _scan_checkpoints() -> List[ModelCheckpoint]:
    """Scan src/run__*/ and src/outputs/ directories for all .ckpt files."""
    checkpoints = []
    search_paths = list(_SRC.glob("run__*")) + ([OUTPUTS_DIR] if OUTPUTS_DIR.exists() else [])

    for base in search_paths:
        for ckpt_path in sorted(base.rglob("*.ckpt")):
            try:
                stat = ckpt_path.stat()
                size_mb = stat.st_size / (1024 * 1024)
                created_at = datetime.fromtimestamp(stat.st_mtime).isoformat()

                parts = ckpt_path.parts
                run_id = ckpt_path.parent.name
                is_best = "best_model" in ckpt_path.name
                checkpoint_id = str(abs(hash(str(ckpt_path))) % 10**8).zfill(8)

                checkpoints.append(ModelCheckpoint(
                    checkpoint_id=checkpoint_id,
                    run_id=run_id,
                    path=str(ckpt_path),
                    created_at=created_at,
                    is_best=is_best,
                    size_mb=round(size_mb, 2),
                ))
            except Exception:
                continue

    return checkpoints


@router.get("", response_model=List[ModelCheckpoint])
def list_models():
    """List all available model checkpoints found in the outputs directory."""
    return _scan_checkpoints()


@router.get("/active")
def get_active_model():
    """Get the currently loaded model checkpoint."""
    active_id = inf_svc.get_active_model_id()
    if active_id is None:
        return {"active": False, "message": "No model currently loaded. Using demo mode."}
    return {"active": True, "checkpoint_id": active_id}


@router.post("/{checkpoint_id}/load", response_model=ModelLoadResponse)
def load_model(checkpoint_id: str):
    """Load a model checkpoint for inference."""
    checkpoints = _scan_checkpoints()
    match = next((c for c in checkpoints if c.checkpoint_id == checkpoint_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Checkpoint {checkpoint_id} not found")

    success = inf_svc.load_model(match.path, checkpoint_id)
    if not success:
        return ModelLoadResponse(
            checkpoint_id=checkpoint_id,
            loaded=False,
            message="Failed to load checkpoint (model may require GPU or missing dependencies). Falling back to demo mode.",
        )
    return ModelLoadResponse(
        checkpoint_id=checkpoint_id,
        loaded=True,
        message=f"Checkpoint {checkpoint_id} loaded successfully.",
    )


@router.post("/unload")
def unload_model():
    """Unload the currently loaded model (revert to demo mode)."""
    inf_svc.unload_model()
    return {"message": "Model unloaded. Inference will use demo mode."}


@router.get("/{checkpoint_id}")
def get_model_info(checkpoint_id: str):
    """Get details about a specific checkpoint."""
    checkpoints = _scan_checkpoints()
    match = next((c for c in checkpoints if c.checkpoint_id == checkpoint_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Checkpoint {checkpoint_id} not found")
    return match
