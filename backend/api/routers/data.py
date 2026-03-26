import io
import json
import uuid
import numpy as np
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List

from ..schemas import ECGUploadResponse, ECGWaveformData, TabularUploadResponse, DataSample

router = APIRouter(prefix="/data", tags=["Data"])

UPLOADS_DIR = Path(__file__).parent.parent / "uploads"
ECG_DIR = UPLOADS_DIR / "ecg"
TAB_DIR = UPLOADS_DIR / "tabular"
META_FILE = UPLOADS_DIR / "metadata.json"

ECG_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# ── Demo dataset paths ────────────────────────────────────────────────────────
_SRC = Path(__file__).parent.parent.parent / "src"
_MEMMAP_DIR = _SRC / "data" / "memmap"


def _load_demo_ecg(sample_idx: int) -> np.ndarray:
    """Load ECG for a demo sample from the memmap. Returns (seq_len, 12) array."""
    meta_path = _MEMMAP_DIR / "memmap_meta.npz"
    memmap_path = _MEMMAP_DIR / "memmap.npy"
    if not meta_path.exists() or not memmap_path.exists():
        return None
    meta = np.load(meta_path, allow_pickle=True)
    n = len(meta["start"])
    idx = sample_idx % n
    start = int(meta["start"][idx])
    length = int(meta["length"][idx])
    shape = tuple(meta["shape"][int(meta["file_idx"][idx])])
    mm = np.memmap(memmap_path, dtype=np.float32, mode="r", shape=shape)
    ecg = np.array(mm[start: start + length])  # (seq_len, 12)
    return ecg


def _load_demo_tabular(sample_idx: int) -> np.ndarray:
    """Load tabular features for a demo sample."""
    parquet_path = _MEMMAP_DIR / "mds_icu_preprocessed2_ecg.parquet"
    if not parquet_path.exists():
        return np.zeros(1602, dtype=np.float32)
    import pandas as pd
    df = pd.read_parquet(parquet_path, columns=["features"])
    n = len(df)
    feat = df["features"].iloc[sample_idx % n]
    return np.array(feat, dtype=np.float32)


def _load_meta() -> dict:
    if META_FILE.exists():
        return json.loads(META_FILE.read_text())
    return {"ecg": {}, "tabular": {}}


def _save_meta(meta: dict):
    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps(meta, indent=2))


@router.post("/upload/ecg", response_model=ECGUploadResponse)
async def upload_ecg(file: UploadFile = File(...)):
    """
    Upload an ECG waveform file (.npy or .csv).
    Expected shape: (12, N) where 12=leads, N=samples.
    """
    ECG_DIR.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    file_id = str(uuid.uuid4())[:8]

    # Parse
    try:
        if file.filename.endswith(".npy"):
            arr = np.load(io.BytesIO(content), allow_pickle=False).astype(np.float32)
        elif file.filename.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(io.StringIO(content.decode()))
            arr = df.values.T.astype(np.float32)
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use .npy or .csv")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Parse error: {e}")

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]  # single-channel

    channels, length = arr.shape[0], arr.shape[-1]
    fs = 100.0
    duration = length / fs

    # Save
    save_path = ECG_DIR / f"{file_id}.npy"
    np.save(save_path, arr)

    meta = _load_meta()
    meta["ecg"][file_id] = {
        "file_id": file_id,
        "filename": file.filename,
        "channels": channels,
        "length": length,
        "fs": fs,
        "duration_seconds": duration,
        "path": str(save_path),
        "uploaded_at": datetime.utcnow().isoformat(),
    }
    _save_meta(meta)

    return ECGUploadResponse(
        file_id=file_id,
        filename=file.filename,
        channels=channels,
        length=length,
        fs=fs,
        duration_seconds=duration,
        message=f"ECG uploaded successfully. {channels} channels × {length} samples at {fs}Hz",
    )


@router.get("/ecg", response_model=List[dict])
def list_ecg_files():
    """List all uploaded ECG files."""
    meta = _load_meta()
    return list(meta.get("ecg", {}).values())


@router.get("/ecg/{file_id}", response_model=ECGWaveformData)
def get_ecg_waveform(file_id: str, max_points: int = 1000):
    """
    Get ECG waveform data for visualization.
    Returns downsampled data (max_points per channel) for performance.
    """
    meta = _load_meta()
    ecg_meta = meta.get("ecg", {}).get(file_id)
    if not ecg_meta:
        raise HTTPException(status_code=404, detail=f"ECG file {file_id} not found")

    arr = np.load(ecg_meta["path"])
    channels, length = arr.shape[0], arr.shape[-1]
    fs = ecg_meta["fs"]

    # Downsample if needed
    if length > max_points:
        indices = np.linspace(0, length - 1, max_points, dtype=int)
        arr_ds = arr[:, indices]
        time_axis = (indices / fs).tolist()
    else:
        arr_ds = arr
        time_axis = (np.arange(length) / fs).tolist()

    lead_names = ECG_LEAD_NAMES[:channels] if channels <= 12 else [f"Ch{i+1}" for i in range(channels)]
    data = {lead_names[i]: arr_ds[i].tolist() for i in range(channels)}

    return ECGWaveformData(
        file_id=file_id,
        channels=channels,
        length=length,
        fs=fs,
        duration_seconds=ecg_meta["duration_seconds"],
        lead_names=lead_names,
        data=data,
        time_axis=time_axis,
    )


@router.get("/ecg/demo/{lead_config}")
def get_demo_ecg(lead_config: str = "12lead", seed: int = 42, sample_idx: int = 0):
    """
    Get a demo ECG for visualization.
    Serves real data from the trained demo dataset when available.
    lead_config: '12lead', 'single', 'bipolar'
    """
    fs = 100.0

    # Try real data from memmap first
    real_ecg = _load_demo_ecg(sample_idx if seed == 42 else seed % 300)
    if real_ecg is not None:
        # real_ecg: (seq_len, 12)
        ecg_T = real_ecg.T  # (12, seq_len)
        N = ecg_T.shape[1]
        n_channels = {"12lead": 12, "single": 1, "bipolar": 2}.get(lead_config, 12)
        n_channels = min(n_channels, ecg_T.shape[0])
        ecg_out = ecg_T[:n_channels]
        source = "real_demo"
    else:
        # Fallback: generate synthetic
        np.random.seed(seed)
        duration = 10.0
        N = int(duration * fs)
        t = np.linspace(0, duration, N)
        n_channels = {"12lead": 12, "single": 1, "bipolar": 2}.get(lead_config, 12)
        ecg_out = np.zeros((n_channels, N), dtype=np.float32)
        heart_rate = 72 + np.random.randint(-15, 15)
        period = 60.0 / heart_rate
        for ch in range(n_channels):
            amplitude = np.random.uniform(0.7, 1.3)
            for beat_t in np.arange(0.1, duration, period):
                p_t = beat_t - 0.16
                if p_t > 0:
                    ecg_out[ch] += 0.1 * amplitude * np.exp(-((t - p_t) ** 2) / (2 * 0.02 ** 2))
                ecg_out[ch] -= 0.05 * amplitude * np.exp(-((t - beat_t + 0.02) ** 2) / (2 * 0.005 ** 2))
                ecg_out[ch] += amplitude * np.exp(-((t - beat_t) ** 2) / (2 * 0.008 ** 2))
                ecg_out[ch] -= 0.03 * amplitude * np.exp(-((t - beat_t - 0.02) ** 2) / (2 * 0.005 ** 2))
                t_t = beat_t + 0.2
                ecg_out[ch] += 0.2 * amplitude * np.exp(-((t - t_t) ** 2) / (2 * 0.04 ** 2))
            ecg_out[ch] += np.random.normal(0, 0.03, N).astype(np.float32)
        source = "synthetic"

    lead_names = ECG_LEAD_NAMES[:n_channels]
    duration = N / fs

    # Downsample for response (max 2000 points per channel)
    max_pts = 2000
    if N > max_pts:
        idx = np.linspace(0, N - 1, max_pts, dtype=int)
        time_axis = (idx / fs).tolist()
        data = {lead_names[i]: ecg_out[i][idx].tolist() for i in range(n_channels)}
    else:
        time_axis = (np.arange(N) / fs).tolist()
        data = {lead_names[i]: ecg_out[i].tolist() for i in range(n_channels)}

    return {
        "file_id": f"demo_{lead_config}_{seed}",
        "channels": n_channels,
        "length": N,
        "fs": fs,
        "duration_seconds": duration,
        "lead_names": lead_names,
        "data": data,
        "time_axis": time_axis,
        "source": source,
    }


@router.post("/upload/tabular", response_model=TabularUploadResponse)
async def upload_tabular(file: UploadFile = File(...)):
    """
    Upload a tabular features file (.npy, .json, or .csv).
    Expected: 1602 continuous clinical features.
    """
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    file_id = str(uuid.uuid4())[:8]

    try:
        if file.filename.endswith(".npy"):
            arr = np.load(io.BytesIO(content), allow_pickle=False).astype(np.float32).flatten()
        elif file.filename.endswith(".json"):
            d = json.loads(content)
            vals = list(d.values()) if isinstance(d, dict) else d
            arr = np.array(vals, dtype=np.float32)
        elif file.filename.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(io.StringIO(content.decode()))
            arr = df.values.flatten().astype(np.float32)
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use .npy, .json, or .csv")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Parse error: {e}")

    save_path = TAB_DIR / f"{file_id}.npy"
    np.save(save_path, arr)

    meta = _load_meta()
    meta["tabular"][file_id] = {
        "file_id": file_id,
        "filename": file.filename,
        "num_features": len(arr),
        "path": str(save_path),
        "uploaded_at": datetime.utcnow().isoformat(),
    }
    _save_meta(meta)

    return TabularUploadResponse(
        file_id=file_id,
        filename=file.filename,
        num_features=len(arr),
        message=f"Tabular data uploaded. {len(arr)} features.",
    )


@router.get("/tabular", response_model=List[dict])
def list_tabular_files():
    """List all uploaded tabular feature files."""
    meta = _load_meta()
    return list(meta.get("tabular", {}).values())


@router.get("/tabular/{file_id}")
def get_tabular_stats(file_id: str):
    """Get statistics for an uploaded tabular feature file."""
    meta = _load_meta()
    tab_meta = meta.get("tabular", {}).get(file_id)
    if not tab_meta:
        raise HTTPException(status_code=404, detail=f"Tabular file {file_id} not found")

    arr = np.load(tab_meta["path"])
    return {
        **tab_meta,
        "stats": {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "nan_count": int(np.isnan(arr).sum()),
            "nonzero_count": int(np.count_nonzero(arr)),
        },
    }


@router.get("/samples", response_model=List[DataSample])
def list_samples():
    """List available demo/sample datasets from trained demo dataset."""
    has_real = (_MEMMAP_DIR / "memmap.npy").exists()
    samples = []
    descriptions = [
        "ICU Patient #1 — Normal sinus rhythm, stable vitals",
        "ICU Patient #2 — Elevated heart rate, respiratory monitoring",
        "ICU Patient #3 — Post-cardiac procedure, medication review",
        "ICU Patient #4 — Sepsis screening, multi-organ assessment",
        "ICU Patient #5 — Bradycardia, electrolyte monitoring",
    ]
    for i in range(5):
        samples.append(DataSample(
            sample_id=f"demo_{i}",
            description=descriptions[i] + (" (real demo data)" if has_real else " (synthetic)"),
            ecg_channels=12,
            ecg_duration_s=10.0,
            num_tabular_features=1602,
            has_labels=has_real,
        ))
    return samples


@router.get("/ecg/demo_sample/{sample_id}")
def get_demo_sample_ecg(sample_id: str, max_points: int = 2000):
    """Get ECG waveform for a specific demo sample by ID (e.g. 'demo_0')."""
    try:
        idx = int(sample_id.replace("demo_", ""))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid sample_id format, use 'demo_N'")

    ecg = _load_demo_ecg(idx)
    if ecg is None:
        raise HTTPException(status_code=404, detail="Demo dataset not found. Run train_demo.sh first.")

    # ecg: (seq_len, 12) → transpose to (12, seq_len)
    ecg_T = ecg.T
    n_channels, N = ecg_T.shape
    fs = 100.0

    if N > max_points:
        idxs = np.linspace(0, N - 1, max_points, dtype=int)
        time_axis = (idxs / fs).tolist()
        data = {ECG_LEAD_NAMES[i]: ecg_T[i][idxs].tolist() for i in range(n_channels)}
    else:
        time_axis = (np.arange(N) / fs).tolist()
        data = {ECG_LEAD_NAMES[i]: ecg_T[i].tolist() for i in range(n_channels)}

    return {
        "sample_id": sample_id,
        "file_id": sample_id,
        "channels": n_channels,
        "length": N,
        "fs": fs,
        "duration_seconds": N / fs,
        "lead_names": ECG_LEAD_NAMES[:n_channels],
        "data": data,
        "time_axis": time_axis,
        "source": "real_demo",
    }
