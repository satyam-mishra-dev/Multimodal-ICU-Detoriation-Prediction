"""Patients router — clinical patient roster + per-patient predictions."""
from fastapi import APIRouter, HTTPException
from typing import List
import numpy as np

from ..patient_profiles import PATIENTS, PATIENT_MAP, get_patient_ecg_and_tabular, LEAD_NAMES, FS
from ..services import inference_service as svc

router = APIRouter(prefix="/patients", tags=["Patients"])


@router.get("")
def list_patients():
    """List all ICU patients with demographic and clinical context."""
    return [
        {
            "patient_id": p.patient_id,
            "name": p.name,
            "age": p.age,
            "gender": p.gender,
            "room": p.room,
            "diagnosis": p.diagnosis,
            "history": p.history,
            "ecg_summary": p.ecg_summary,
            "vitals": {
                "hr": p.vitals.hr,
                "sbp": p.vitals.sbp,
                "dbp": p.vitals.dbp,
                "spo2": p.vitals.spo2,
                "rr": p.vitals.rr,
                "temp": p.vitals.temp,
                "gcs": p.vitals.gcs,
                "map": p.vitals.map_,
            },
            "labs": p.labs,
            "risk_narrative": p.risk_narrative,
        }
        for p in PATIENTS
    ]


@router.post("/{patient_id}/predict")
def predict_patient(patient_id: str):
    """
    Run the trained model on a specific patient's ECG + tabular features.
    Returns prediction results alongside model AUROC scores and patient context.
    """
    if patient_id not in PATIENT_MAP:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    profile = PATIENT_MAP[patient_id]
    ecg, tabular = get_patient_ecg_and_tabular(patient_id)
    prediction = svc.run_prediction(ecg, tabular)

    # Load model scores
    import pickle
    from pathlib import Path
    _src = Path(__file__).parent.parent.parent / "src"
    scores_files = sorted(_src.glob("run__*/scores_test.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    per_label_auroc = {}
    macro_auroc = None
    run_id = None
    if scores_files:
        with open(scores_files[0], "rb") as f:
            scores = pickle.load(f)
        run_id = scores_files[0].parent.name
        macro_auroc = scores.get("test_auroc_macro_agg0")
        for k, v in scores.items():
            if k.startswith("test_auroc_") and k.endswith("_agg0"):
                lbl = k.replace("test_auroc_", "").replace("_agg0", "")
                if lbl not in ("micro", "macro"):
                    per_label_auroc[lbl] = round(float(v), 4) if v is not None else None

    return {
        "patient": {
            "patient_id": profile.patient_id,
            "name": profile.name,
            "age": profile.age,
            "gender": profile.gender,
            "room": profile.room,
            "diagnosis": profile.diagnosis,
            "history": profile.history,
            "ecg_summary": profile.ecg_summary,
            "risk_narrative": profile.risk_narrative,
            "vitals": {
                "hr": profile.vitals.hr,
                "sbp": profile.vitals.sbp,
                "dbp": profile.vitals.dbp,
                "spo2": profile.vitals.spo2,
                "rr": profile.vitals.rr,
                "temp": profile.vitals.temp,
                "gcs": profile.vitals.gcs,
                "map": profile.vitals.map_,
            },
            "labs": profile.labs,
        },
        "prediction": {
            "prediction_id": prediction.prediction_id,
            "demo_mode": prediction.demo_mode,
            "model_used": prediction.model_used,
            "processing_time_ms": prediction.processing_time_ms,
            "ecg_channels": prediction.ecg_channels,
            "ecg_length": prediction.ecg_length,
            "tabular_features": prediction.tabular_features,
            "labels": [
                {
                    "label": l.label,
                    "group": l.group,
                    "probability": l.probability,
                    "confidence_low": l.confidence_low,
                    "confidence_high": l.confidence_high,
                }
                for l in prediction.labels
            ],
        },
        "model_performance": {
            "run_id": run_id,
            "macro_auroc": round(float(macro_auroc), 4) if macro_auroc is not None else None,
            "per_label_auroc": per_label_auroc,
        },
    }


@router.get("/{patient_id}/ecg")
def get_patient_ecg(patient_id: str, max_points: int = 1000):
    """Return ECG waveform for visualization."""
    if patient_id not in PATIENT_MAP:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    profile = PATIENT_MAP[patient_id]
    ecg, _ = get_patient_ecg_and_tabular(patient_id)
    n_ch, N = ecg.shape

    if N > max_points:
        idxs = np.linspace(0, N - 1, max_points, dtype=int)
        time_axis = (idxs / FS).tolist()
        data = {LEAD_NAMES[i]: ecg[i][idxs].tolist() for i in range(n_ch)}
    else:
        time_axis = (np.arange(N) / FS).tolist()
        data = {LEAD_NAMES[i]: ecg[i].tolist() for i in range(n_ch)}

    return {
        "patient_id": patient_id,
        "file_id": patient_id,
        "channels": n_ch,
        "length": N,
        "fs": FS,
        "duration_seconds": N / FS,
        "lead_names": LEAD_NAMES[:n_ch],
        "data": data,
        "time_axis": time_axis,
        "ecg_type": profile.ecg_type,
        "ecg_summary": profile.ecg_summary,
    }
