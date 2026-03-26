"""
Training service: manages background training processes,
persists run state to disk, streams logs.
"""
import asyncio
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from ..schemas import TrainingConfig, TrainingRun, TrainingStatus

RUNS_FILE = Path(__file__).parent.parent / "state" / "runs.json"
BACKEND_SRC = Path(__file__).parent.parent.parent / "src"


def _now() -> str:
    return datetime.utcnow().isoformat()


def _load_runs() -> Dict[str, dict]:
    if RUNS_FILE.exists():
        return json.loads(RUNS_FILE.read_text())
    return {}


def _save_runs(runs: Dict[str, dict]):
    RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUNS_FILE.write_text(json.dumps(runs, indent=2))


def _build_overrides(cfg: TrainingConfig) -> list[str]:
    """Convert TrainingConfig → Hydra override strings."""
    overrides = [
        f"base.batch_size={cfg.batch_size}",
        f"base.lr=[{cfg.learning_rate}]",
        f"base.weight_decay={cfg.weight_decay}",
        f"trainer.epochs={cfg.epochs}",
        f"trainer.gpus={cfg.gpus}",
        f"trainer.precision={cfg.precision}",
        f"trainer.metadata={cfg.metadata}",
        f"ts0.pred.model_dim={cfg.model_dim}",
        f"ts0.pred.state_dim={cfg.state_dim}",
    ]
    if cfg.resume:
        overrides.append(f"trainer.resume={cfg.resume}")
    if cfg.eval_only:
        overrides.append(f"trainer.eval_only={cfg.eval_only}")
    return overrides


async def start_training(cfg: TrainingConfig) -> str:
    run_id = str(uuid.uuid4())[:8]
    runs = _load_runs()

    runs[run_id] = {
        "run_id": run_id,
        "status": TrainingStatus.PENDING.value,
        "config": cfg.model_dump(),
        "created_at": _now(),
        "updated_at": _now(),
        "pid": None,
        "logs": [],
        "metrics": {},
    }
    _save_runs(runs)

    asyncio.create_task(_run_training(run_id, cfg))
    return run_id


async def _run_training(run_id: str, cfg: TrainingConfig):
    runs = _load_runs()
    runs[run_id]["status"] = TrainingStatus.RUNNING.value
    runs[run_id]["updated_at"] = _now()
    _save_runs(runs)

    overrides = _build_overrides(cfg)
    cmd = [
        sys.executable, "main_all.py",
        "--config-name", "config_supervised_multimodal_mdsicu",
    ] + overrides

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(BACKEND_SRC),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(BACKEND_SRC)},
        )

        runs = _load_runs()
        runs[run_id]["pid"] = proc.pid
        _save_runs(runs)

        # Stream logs
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip()
            runs = _load_runs()
            runs[run_id]["logs"].append(decoded)
            # Keep last 500 lines only
            if len(runs[run_id]["logs"]) > 500:
                runs[run_id]["logs"] = runs[run_id]["logs"][-500:]
            _save_runs(runs)

        await proc.wait()

        runs = _load_runs()
        if proc.returncode == 0:
            runs[run_id]["status"] = TrainingStatus.COMPLETED.value
        else:
            runs[run_id]["status"] = TrainingStatus.FAILED.value
        runs[run_id]["updated_at"] = _now()
        _save_runs(runs)

    except Exception as e:
        runs = _load_runs()
        runs[run_id]["status"] = TrainingStatus.FAILED.value
        runs[run_id]["logs"].append(f"[ERROR] {str(e)}")
        runs[run_id]["updated_at"] = _now()
        _save_runs(runs)


def get_all_runs() -> list[dict]:
    return list(_load_runs().values())


def get_run(run_id: str) -> Optional[dict]:
    return _load_runs().get(run_id)


def stop_run(run_id: str) -> bool:
    runs = _load_runs()
    run = runs.get(run_id)
    if not run:
        return False
    pid = run.get("pid")
    if pid:
        try:
            import signal
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    runs[run_id]["status"] = TrainingStatus.STOPPED.value
    runs[run_id]["updated_at"] = _now()
    _save_runs(runs)
    return True


def delete_run(run_id: str) -> bool:
    runs = _load_runs()
    if run_id not in runs:
        return False
    del runs[run_id]
    _save_runs(runs)
    return True
