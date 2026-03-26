from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List
import asyncio

from ..schemas import (
    TrainingConfig, TrainingStartResponse, TrainingRun,
    TrainingRunSummary, TrainingStatus
)
from ..services import training_service as svc

router = APIRouter(prefix="/training", tags=["Training"])


@router.post("/start", response_model=TrainingStartResponse)
async def start_training(cfg: TrainingConfig):
    """Start a new training run with the given configuration."""
    run_id = await svc.start_training(cfg)
    return TrainingStartResponse(
        run_id=run_id,
        status=TrainingStatus.PENDING,
        message=f"Training run {run_id} started. Monitor via GET /training/runs/{run_id}",
    )


@router.get("/runs", response_model=List[TrainingRunSummary])
def list_runs():
    """List all training runs (summary)."""
    runs = svc.get_all_runs()
    summaries = []
    for r in sorted(runs, key=lambda x: x["created_at"], reverse=True):
        metrics = r.get("metrics", {})
        summaries.append(TrainingRunSummary(
            run_id=r["run_id"],
            status=r["status"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            epochs_done=metrics.get("epochs_done", 0),
            best_auroc=metrics.get("best_auroc"),
        ))
    return summaries


@router.get("/runs/{run_id}", response_model=TrainingRun)
def get_run(run_id: str):
    """Get full details of a specific training run."""
    run = svc.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return TrainingRun(**run)


@router.get("/runs/{run_id}/logs")
def get_run_logs(run_id: str, tail: int = 100):
    """Get the last N log lines from a training run."""
    run = svc.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    logs = run.get("logs", [])
    return {"run_id": run_id, "logs": logs[-tail:], "total_lines": len(logs)}


@router.get("/runs/{run_id}/logs/stream")
async def stream_logs(run_id: str):
    """Stream logs from a running training job via SSE."""
    run = svc.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    sent_idx = 0

    async def event_generator():
        nonlocal sent_idx
        while True:
            current = svc.get_run(run_id)
            if not current:
                break
            logs = current.get("logs", [])
            new_logs = logs[sent_idx:]
            for line in new_logs:
                yield f"data: {line}\n\n"
            sent_idx += len(new_logs)

            status = current.get("status", "")
            if status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.STOPPED):
                yield f"data: [DONE] Status: {status}\n\n"
                break

            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/runs/{run_id}/stop")
def stop_run(run_id: str):
    """Stop a running training job."""
    success = svc.stop_run(run_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return {"message": f"Run {run_id} stop signal sent"}


@router.delete("/runs/{run_id}")
def delete_run(run_id: str):
    """Delete a training run record."""
    run = svc.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if run["status"] == TrainingStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Cannot delete a running job. Stop it first.")
    svc.delete_run(run_id)
    return {"message": f"Run {run_id} deleted"}
