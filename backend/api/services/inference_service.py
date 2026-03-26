"""
Inference service: loads trained Lightning checkpoints and runs real predictions.
Falls back to demo mode when no model is loaded.
"""
import sys
import time
import uuid
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

from ..schemas import PredictionLabel, PredictionResponse

# ── Paths ────────────────────────────────────────────────────────────────────
_BACKEND = Path(__file__).parent.parent.parent
_MODELS_DIR = _BACKEND / "models"          # backend/models/ — preferred checkpoint location
_SRC = _BACKEND / "src"
_DATA = _SRC / "data" / "memmap"

# ── Load label names from demo dataset ───────────────────────────────────────
def _load_label_names():
    pkl = _DATA / "lbl_itos.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            lbl = pickle.load(f)
        if isinstance(lbl, list):
            return lbl
        if isinstance(lbl, dict):
            return [lbl[i] for i in sorted(lbl)]
    # fallback
    return [f"label_{i}" for i in range(33)]

LABEL_NAMES = _load_label_names()

# Map each label to a group for display
def _label_to_group(lbl: str) -> str:
    if lbl.startswith("mortality"):
        return "Mortality"
    if lbl.startswith("med_"):
        return "Medication"
    if lbl.startswith("detr_"):
        return "Deterioration"
    if lbl.startswith("od_"):
        return "Organ Dysfunction"
    return "Other"

LABEL_GROUPS = {
    "Mortality": [],
    "Medication": [],
    "Deterioration": [],
    "Organ Dysfunction": [],
    "Other": [],
}
for _lbl in LABEL_NAMES:
    LABEL_GROUPS[_label_to_group(_lbl)].append(_lbl)

ALL_LABELS = [(lbl, _label_to_group(lbl)) for lbl in LABEL_NAMES]

# ── Global model state ────────────────────────────────────────────────────────
_loaded_model = None
_loaded_checkpoint_path: Optional[str] = None
_loaded_checkpoint_id: Optional[str] = None

# ── Auto-discover latest checkpoint ──────────────────────────────────────────
def find_latest_checkpoint() -> Optional[Path]:
    """
    Prefer backend/models/best_model.ckpt.
    Fall back to latest src/run__*/best_model.ckpt if models/ is empty.
    """
    # 1. Preferred: backend/models/
    models_ckpt = _MODELS_DIR / "best_model.ckpt"
    if models_ckpt.exists():
        return models_ckpt

    # 2. Fallback: latest run directory
    run_ckpts = sorted(
        _SRC.glob("run__*/best_model.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return run_ckpts[0] if run_ckpts else None


def ensure_model_exists() -> bool:
    """
    Called at startup. If backend/models/ is empty, copy the latest checkpoint there.
    If no checkpoint exists anywhere, trigger training.
    Returns True if a model is available after the call.
    """
    import shutil
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models_ckpt = _MODELS_DIR / "best_model.ckpt"
    if models_ckpt.exists():
        return True

    # Try to copy from latest run
    run_ckpts = sorted(
        _SRC.glob("run__*/best_model.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if run_ckpts:
        shutil.copy2(str(run_ckpts[0]), str(models_ckpt))
        print(f"[startup] Copied checkpoint {run_ckpts[0]} → {models_ckpt}")
        return True

    # No checkpoint found — trigger training in background
    import subprocess, threading
    train_script = _BACKEND / "train_demo.sh"
    if train_script.exists():
        print("[startup] No model found in backend/models/ — launching training…")
        def _train():
            result = subprocess.run(["bash", str(train_script)], cwd=str(_BACKEND))
            if result.returncode == 0:
                # After training, copy best checkpoint
                new_ckpts = sorted(
                    _SRC.glob("run__*/best_model.ckpt"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if new_ckpts:
                    shutil.copy2(str(new_ckpts[0]), str(models_ckpt))
                    print(f"[startup] Training complete. Model saved to {models_ckpt}")
        threading.Thread(target=_train, daemon=True).start()
    return False


def _load_lightning_model(checkpoint_path: Path):
    """Load a trained MultimodalModel from a Lightning checkpoint."""
    import os
    import torch
    import functools

    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    # Ensure weights_only=False for our trusted checkpoints
    orig_load = torch.load
    @functools.wraps(orig_load)
    def _safe_load(*a, **kw):
        kw['weights_only'] = False
        return orig_load(*a, **kw)

    torch.load = _safe_load
    orig_cwd = os.getcwd()
    try:
        # The model's __init__ loads dataset files via relative paths anchored at src/
        os.chdir(str(_SRC))
        from clinical_ts.task.multimodal import MultimodalModel
        model = MultimodalModel.load_from_checkpoint(
            str(checkpoint_path), map_location="cpu"
        )
        model.eval()
    finally:
        os.chdir(orig_cwd)
        torch.load = orig_load

    return model


def get_active_model_id() -> Optional[str]:
    return _loaded_checkpoint_id


# ── Prediction helpers ────────────────────────────────────────────────────────
class _MockTrainer:
    """Minimal trainer stub so model.forward() can read trainer attributes."""
    accumulate_grad_batches = 1
    num_devices = 1
    global_step = 0


def _attach_mock_trainer(model) -> None:
    """Attach a stub trainer so model.forward() can access trainer attributes without a Trainer."""
    if getattr(model, "_trainer", None) is None:
        model._trainer = _MockTrainer()


def load_model(checkpoint_path: str, checkpoint_id: str) -> bool:
    global _loaded_model, _loaded_checkpoint_path, _loaded_checkpoint_id
    try:
        m = _load_lightning_model(Path(checkpoint_path))
        _attach_mock_trainer(m)
        _loaded_model = m
        _loaded_checkpoint_path = checkpoint_path
        _loaded_checkpoint_id = checkpoint_id
        return True
    except Exception as e:
        print(f"[inference] model load failed: {e}")
        return False


def unload_model():
    global _loaded_model, _loaded_checkpoint_path, _loaded_checkpoint_id
    _loaded_model = None
    _loaded_checkpoint_path = None
    _loaded_checkpoint_id = None


def _auto_load():
    """Auto-load the latest checkpoint on first call."""
    global _loaded_model, _loaded_checkpoint_path, _loaded_checkpoint_id
    if _loaded_model is not None:
        return
    ckpt = find_latest_checkpoint()
    if ckpt is None:
        return
    try:
        m = _load_lightning_model(ckpt)
        _attach_mock_trainer(m)
        _loaded_model = m
        _loaded_checkpoint_path = str(ckpt)
        _loaded_checkpoint_id = ckpt.parent.name
        print(f"[inference] auto-loaded model from {ckpt}")
    except Exception as e:
        print(f"[inference] auto-load failed: {e}")


def _real_predict(ecg: np.ndarray, tabular: np.ndarray) -> list[PredictionLabel]:
    """Run the actual model forward pass."""
    import torch

    model = _loaded_model
    model.eval()

    # Model expects channels-first: (B, C, L)
    if ecg.ndim == 1:
        ecg = ecg[np.newaxis, :]  # (1, L)
    if ecg.shape[0] != 12:
        ecg = ecg.T  # (L, 12) → (12, L)
    ecg_t = torch.tensor(ecg[np.newaxis], dtype=torch.float32)  # (1, 12, L)

    tab_t = torch.tensor(tabular[np.newaxis], dtype=torch.float32)  # (1, 1602)

    with torch.no_grad():
        out = model.forward(seq=ecg_t, label=None, static=tab_t)
        # forward returns a dict; predictions are under "input_predicted"
        if isinstance(out, dict):
            logits = out.get("input_predicted", out.get("logits", out.get("seq", next(iter(out.values())))))
        else:
            logits = out
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    labels = []
    for i, (name, group) in enumerate(ALL_LABELS):
        p = float(np.clip(probs[i] if i < len(probs) else 0.5, 0, 1))
        margin = 0.05
        labels.append(PredictionLabel(
            label=name,
            group=group,
            probability=round(p, 4),
            confidence_low=round(max(0.0, p - margin), 4),
            confidence_high=round(min(1.0, p + margin), 4),
        ))
    return sorted(labels, key=lambda x: x.probability, reverse=True)


def _demo_predict(ecg: np.ndarray, tabular: Optional[np.ndarray]) -> list[PredictionLabel]:
    """Deterministic synthetic predictions seeded by ECG content."""
    seed = int(np.abs(ecg).mean() * 1e6) % (2**31)
    rng = np.random.default_rng(seed)
    labels = []
    for name, group in ALL_LABELS:
        p = float(rng.beta(1.5, 5))
        margin = float(rng.uniform(0.03, 0.08))
        labels.append(PredictionLabel(
            label=name,
            group=group,
            probability=round(p, 4),
            confidence_low=round(max(0.0, p - margin), 4),
            confidence_high=round(min(1.0, p + margin), 4),
        ))
    return sorted(labels, key=lambda x: x.probability, reverse=True)


# ── Public API ────────────────────────────────────────────────────────────────
def run_prediction(
    ecg_data: np.ndarray,
    tabular_data: Optional[np.ndarray] = None,
) -> PredictionResponse:
    _auto_load()

    start = time.perf_counter()
    prediction_id = str(uuid.uuid4())[:8]

    if tabular_data is None:
        tabular_data = np.zeros(1602, dtype=np.float32)

    demo_mode = True
    model_used = None

    if _loaded_model is not None:
        try:
            labels = _real_predict(ecg_data, tabular_data)
            demo_mode = False
            model_used = _loaded_checkpoint_id
        except Exception as e:
            print(f"[inference] real predict failed ({e}), using demo fallback")
            labels = _demo_predict(ecg_data, tabular_data)
    else:
        labels = _demo_predict(ecg_data, tabular_data)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        prediction_id=prediction_id,
        labels=labels,
        ecg_channels=int(ecg_data.shape[0]) if ecg_data.ndim > 1 else 1,
        ecg_length=int(ecg_data.shape[-1]),
        tabular_features=int(len(tabular_data)),
        processing_time_ms=round(elapsed_ms, 2),
        model_used=model_used,
        demo_mode=demo_mode,
    )
