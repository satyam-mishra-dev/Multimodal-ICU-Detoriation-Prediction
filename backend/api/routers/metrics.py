import json
import struct
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException

from ..schemas import RunMetrics, MetricPoint
from ..services import training_service as svc

router = APIRouter(prefix="/metrics", tags=["Metrics"])

OUTPUTS_DIR = Path(__file__).parent.parent.parent / "src" / "outputs"


def _read_tfevents(event_file: Path) -> dict:
    """
    Read TensorBoard event file and extract scalar metrics.
    Returns dict: {tag: [(step, value), ...]}
    """
    scalars: dict = {}
    try:
        # Try using tensorboard's event file reader
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(event_file.parent))
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            scalars[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
    except Exception:
        pass
    return scalars


def _find_event_files(run_dir: Path) -> List[Path]:
    """Find all TensorBoard event files under a run directory."""
    return list(run_dir.rglob("events.out.tfevents.*"))


def _get_run_dirs() -> List[Path]:
    """Get all run output directories."""
    if not OUTPUTS_DIR.exists():
        return []
    dirs = []
    for d in OUTPUTS_DIR.rglob("*"):
        if d.is_dir() and any(d.glob("events.out.tfevents.*")):
            dirs.append(d)
    return dirs


@router.get("/runs", response_model=List[dict])
def list_metric_runs():
    """List all training runs that have metric data."""
    runs = []

    # First, check in-memory tracked runs
    for run in svc.get_all_runs():
        runs.append({
            "run_id": run["run_id"],
            "source": "api",
            "status": run["status"],
            "created_at": run["created_at"],
            "has_tensorboard": False,
            "metrics_summary": run.get("metrics", {}),
        })

    # Then scan outputs for TensorBoard dirs
    for run_dir in _get_run_dirs():
        runs.append({
            "run_id": str(run_dir.relative_to(OUTPUTS_DIR)),
            "source": "tensorboard",
            "status": "completed",
            "created_at": None,
            "has_tensorboard": True,
            "metrics_summary": {},
        })

    return runs


@router.get("/runs/{run_id:path}", response_model=RunMetrics)
def get_run_metrics(run_id: str):
    """
    Get detailed metrics for a specific run.
    For API-tracked runs: returns stored metrics.
    For TensorBoard runs: parses event files.
    """
    # Check API-tracked runs
    api_run = svc.get_run(run_id)
    if api_run:
        m = api_run.get("metrics", {})
        return RunMetrics(
            run_id=run_id,
            train_loss=[MetricPoint(step=i, value=v) for i, v in enumerate(m.get("train_loss", []))],
            val_loss=[MetricPoint(step=i, value=v) for i, v in enumerate(m.get("val_loss", []))],
            val_auroc=[MetricPoint(step=i, value=v) for i, v in enumerate(m.get("val_auroc", []))],
            val_aupr=[MetricPoint(step=i, value=v) for i, v in enumerate(m.get("val_aupr", []))],
            per_class_auroc=m.get("per_class_auroc", {}),
            best_val_auroc=m.get("best_auroc"),
            best_epoch=m.get("best_epoch"),
        )

    # Try TensorBoard event files
    target_dir = OUTPUTS_DIR / run_id
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    event_files = _find_event_files(target_dir)
    if not event_files:
        raise HTTPException(status_code=404, detail=f"No event files found for run {run_id}")

    all_scalars: dict = {}
    for ef in event_files:
        scalars = _read_tfevents(ef)
        for tag, events in scalars.items():
            if tag not in all_scalars:
                all_scalars[tag] = events
            else:
                all_scalars[tag].extend(events)

    def _extract(tag_patterns: list) -> List[MetricPoint]:
        for pattern in tag_patterns:
            for key in all_scalars:
                if pattern.lower() in key.lower():
                    return [MetricPoint(step=s, value=v) for s, v in all_scalars[key]]
        return []

    train_loss = _extract(["train_loss", "train/loss"])
    val_loss = _extract(["val_loss", "val/loss"])
    val_auroc = _extract(["val_auroc", "val/auroc", "auroc"])
    val_aupr = _extract(["val_aupr", "val/aupr", "aupr"])

    best_auroc = max((p.value for p in val_auroc), default=None)
    best_epoch = next((p.step for p in val_auroc if p.value == best_auroc), None) if best_auroc else None

    return RunMetrics(
        run_id=run_id,
        train_loss=train_loss,
        val_loss=val_loss,
        val_auroc=val_auroc,
        val_aupr=val_aupr,
        per_class_auroc={},
        best_val_auroc=best_auroc,
        best_epoch=best_epoch,
    )


@router.get("/demo")
def get_demo_metrics():
    """Return realistic synthetic training metrics for visualization demo."""
    import numpy as np
    np.random.seed(0)
    epochs = 20

    train_loss = []
    val_loss = []
    val_auroc = []
    val_aupr = []

    tl = 0.85
    vl = 0.88
    va = 0.55
    vp = 0.30
    for ep in range(epochs):
        tl = max(0.1, tl * np.random.uniform(0.88, 0.95))
        vl = max(0.12, vl * np.random.uniform(0.89, 0.96))
        va = min(0.92, va + np.random.uniform(0.01, 0.03))
        vp = min(0.85, vp + np.random.uniform(0.01, 0.025))
        train_loss.append({"step": ep, "value": round(float(tl + np.random.normal(0, 0.01)), 4)})
        val_loss.append({"step": ep, "value": round(float(vl + np.random.normal(0, 0.01)), 4)})
        val_auroc.append({"step": ep, "value": round(float(va + np.random.normal(0, 0.005)), 4)})
        val_aupr.append({"step": ep, "value": round(float(vp + np.random.normal(0, 0.005)), 4)})

    return {
        "run_id": "demo",
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_auroc": val_auroc,
        "val_aupr": val_aupr,
        "per_class_auroc": {
            "in_hospital_mortality": 0.84,
            "30_day_mortality": 0.81,
            "vasopressor_use": 0.88,
            "mechanical_ventilation": 0.86,
            "acute_kidney_injury": 0.79,
            "septic_shock": 0.83,
            "cardiac_arrest": 0.77,
            "renal_replacement_therapy": 0.85,
        },
        "best_val_auroc": max(p["value"] for p in val_auroc),
        "best_epoch": max(range(epochs), key=lambda i: val_auroc[i]["value"]),
    }
