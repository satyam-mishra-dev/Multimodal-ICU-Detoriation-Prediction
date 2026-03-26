import io
import json
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional

from ..schemas import PredictionResponse, PredictionLabel
from ..services import inference_service as svc

router = APIRouter(prefix="/inference", tags=["Inference"])


def _parse_ecg(file: UploadFile) -> np.ndarray:
    """Parse uploaded ECG file (.npy or .csv) into numpy array."""
    content = file.file.read()
    if file.filename.endswith(".npy"):
        arr = np.load(io.BytesIO(content), allow_pickle=False)
    elif file.filename.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(io.StringIO(content.decode()))
        arr = df.values.T  # shape: (channels, time)
    else:
        # Try npy first, fallback to raw float array
        try:
            arr = np.load(io.BytesIO(content), allow_pickle=False)
        except Exception:
            raise HTTPException(status_code=400, detail="Unsupported ECG format. Use .npy or .csv")
    return arr.astype(np.float32)


def _parse_tabular(file: UploadFile) -> np.ndarray:
    """Parse uploaded tabular feature file (.npy, .json, .csv)."""
    content = file.file.read()
    if file.filename.endswith(".npy"):
        arr = np.load(io.BytesIO(content), allow_pickle=False)
    elif file.filename.endswith(".json"):
        data = json.loads(content)
        arr = np.array(list(data.values()), dtype=np.float32) if isinstance(data, dict) else np.array(data, dtype=np.float32)
    elif file.filename.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(io.StringIO(content.decode()))
        arr = df.values.flatten()
    else:
        raise HTTPException(status_code=400, detail="Unsupported tabular format. Use .npy, .json, or .csv")
    return arr.astype(np.float32)


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    ecg_file: UploadFile = File(..., description="ECG waveform (.npy or .csv). Shape: (12, N) for 12-lead ECG"),
    tabular_file: Optional[UploadFile] = File(None, description="Tabular clinical features (.npy, .json, or .csv)"),
):
    """
    Run inference on uploaded ECG and optional tabular data.
    Returns probability predictions for all 33 clinical outcome labels.
    """
    try:
        ecg_data = _parse_ecg(ecg_file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse ECG file: {e}")

    tabular_data = None
    if tabular_file is not None:
        try:
            tabular_data = _parse_tabular(tabular_file)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to parse tabular file: {e}")

    result = svc.run_prediction(ecg_data, tabular_data)
    return result


@router.post("/predict/demo", response_model=PredictionResponse)
def predict_demo(seed: int = 42, sample_idx: int = 0):
    """
    Run a demo prediction using real ECG + tabular data from the trained demo dataset.
    Falls back to synthetic data if the demo dataset is not available.
    """
    from pathlib import Path as _Path
    import pickle as _pickle
    _src = _Path(__file__).parent.parent.parent / "src"
    _data = _src / "data" / "memmap"

    ecg = None
    tabular = None

    # Try loading real data from memmap
    if (_data / "memmap.npy").exists() and (_data / "memmap_meta.npz").exists():
        try:
            meta = np.load(_data / "memmap_meta.npz", allow_pickle=True)
            idx = sample_idx % len(meta["start"])
            start = int(meta["start"][idx])
            length = int(meta["length"][idx])
            shape = tuple(meta["shape"][int(meta["file_idx"][idx])])
            mm = np.memmap(_data / "memmap.npy", dtype=np.float32, mode="r", shape=shape)
            ecg_seq = np.array(mm[start: start + length])  # (seq_len, 12)
            ecg = ecg_seq.T.astype(np.float32)  # (12, seq_len)
        except Exception as e:
            print(f"[demo] memmap load failed: {e}")

    if ecg is None:
        np.random.seed(seed)
        t = np.linspace(0, 10.0, 1000)
        ecg = np.zeros((12, 1000), dtype=np.float32)
        for ch in range(12):
            heart_rate = 72 + (ch * 3)
            period = 60.0 / heart_rate
            for beat_t in np.arange(0.1, 10.0, period):
                ecg[ch] += np.exp(-((t - beat_t) ** 2) / (2 * 0.008 ** 2))
            ecg[ch] += np.random.normal(0, 0.05, 1000).astype(np.float32)

    if (_data / "mds_icu_preprocessed2_ecg.parquet").exists():
        try:
            import pandas as pd
            df = pd.read_parquet(_data / "mds_icu_preprocessed2_ecg.parquet", columns=["features"])
            tabular = np.array(df["features"].iloc[sample_idx % len(df)], dtype=np.float32)
        except Exception as e:
            print(f"[demo] tabular load failed: {e}")

    if tabular is None:
        tabular = np.zeros(1602, dtype=np.float32)

    return svc.run_prediction(ecg, tabular)


@router.post("/predict/json", response_model=PredictionResponse)
async def predict_json(payload: dict):
    """
    Run inference with ECG and tabular data provided as JSON.

    Expected payload:
    {
      "ecg": [[...], [...], ...],   // 12 lists of floats (channels)
      "tabular": [...]               // list of 1602 floats (optional)
    }
    """
    if "ecg" not in payload:
        raise HTTPException(status_code=422, detail="Missing 'ecg' field")

    try:
        ecg_data = np.array(payload["ecg"], dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid ECG data: {e}")

    tabular_data = None
    if "tabular" in payload:
        try:
            tabular_data = np.array(payload["tabular"], dtype=np.float32)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid tabular data: {e}")

    return svc.run_prediction(ecg_data, tabular_data)


@router.get("/model_scores")
def get_model_scores():
    """Return per-label AUROC from the latest run's test scores, plus macro AUROC."""
    from pathlib import Path as _P
    import pickle as _pickle
    _src = _P(__file__).parent.parent.parent / "src"

    # Find latest scores_test.pkl
    run_dirs = sorted(_src.glob("run__*/scores_test.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        return {"available": False, "macro_auroc": None, "per_label": {}}

    scores_path = run_dirs[0]
    run_id = scores_path.parent.name

    with open(scores_path, "rb") as f:
        scores = _pickle.load(f)

    # Extract per-label AUROC (the _agg0 keys without _low/_high/_nans suffix)
    per_label = {}
    for k, v in scores.items():
        if k.startswith("test_auroc_") and k.endswith("_agg0") and not any(k.endswith(s) for s in ["_low", "_high", "_nans"]):
            label = k.replace("test_auroc_", "").replace("_agg0", "")
            if label not in ("micro", "macro"):
                per_label[label] = round(float(v), 4) if v is not None else None

    macro = scores.get("test_auroc_macro_agg0")

    return {
        "available": True,
        "run_id": run_id,
        "macro_auroc": round(float(macro), 4) if macro is not None else None,
        "per_label": per_label,
    }
