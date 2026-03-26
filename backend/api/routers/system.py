import os
import platform
from pathlib import Path
from fastapi import APIRouter

from ..schemas import SystemInfo
from ..services import inference_service as inf_svc

router = APIRouter(prefix="/system", tags=["System"])

BACKEND_ROOT = Path(__file__).parent.parent.parent


@router.get("/info", response_model=SystemInfo)
def get_system_info():
    """Get system information: GPU, CPU, memory, library versions."""
    import sys

    gpu_available = False
    gpu_count = 0
    gpu_names = []
    cuda_version = None
    torch_version = "not installed"

    try:
        import torch
        torch_version = torch.__version__
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        if gpu_available:
            cuda_version = torch.version.cuda
    except ImportError:
        pass

    try:
        import psutil
        total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        cpu_count = psutil.cpu_count(logical=True)
    except ImportError:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        total_memory_gb = 0.0

    # Count active runs
    from ..services import training_service as svc
    from ..schemas import TrainingStatus
    active_runs = sum(
        1 for r in svc.get_all_runs()
        if r["status"] == TrainingStatus.RUNNING.value
    )

    return SystemInfo(
        gpu_available=gpu_available,
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        cpu_count=cpu_count,
        total_memory_gb=round(total_memory_gb, 1),
        python_version=sys.version.split()[0],
        torch_version=torch_version,
        cuda_version=cuda_version,
        backend_root=str(BACKEND_ROOT),
        active_model=inf_svc.get_active_model_id() or (
            str(inf_svc.find_latest_checkpoint().parent.name) + " (ready)"
            if inf_svc.find_latest_checkpoint() else None
        ),
        active_runs=active_runs,
    )


@router.get("/health")
def health_check():
    return {"status": "ok", "service": "MDS-ICU API"}
