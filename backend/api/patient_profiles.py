"""
ICU patient profiles with clinically distinct ECG morphologies and tabular feature vectors.
Each patient has a unique clinical presentation that produces different model outputs.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

FS = 100.0          # Hz
DURATION = 10.0     # seconds
N = int(FS * DURATION)  # 1000 samples
N_CHANNELS = 12
N_FEATURES = 1602
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


@dataclass
class Vital:
    hr: float           # beats/min
    sbp: float          # systolic BP mmHg
    dbp: float          # diastolic BP mmHg
    spo2: float         # %
    rr: float           # resp rate /min
    temp: float         # °C
    gcs: int            # Glasgow Coma Scale
    map_: float         # mean arterial pressure


@dataclass
class PatientProfile:
    patient_id: str
    name: str
    age: int
    gender: str
    room: str
    diagnosis: str
    history: str
    vitals: Vital
    labs: dict
    ecg_summary: str
    risk_narrative: str
    ecg_type: str       # internal morphology tag
    feature_bias: dict  # sparse overrides for tabular features


# ── ECG generators ────────────────────────────────────────────────────────────

def _t_axis() -> np.ndarray:
    return np.linspace(0, DURATION, N)


def _gaussian(t, center, sigma, amp):
    return amp * np.exp(-((t - center) ** 2) / (2 * sigma ** 2))


def _normal_beat(t, beat_t, amp=1.0, hr=72):
    """Standard PQRST complex."""
    sig = np.zeros_like(t)
    sig += _gaussian(t, beat_t - 0.16, 0.020, 0.10 * amp)   # P wave
    sig -= _gaussian(t, beat_t - 0.02, 0.005, 0.05 * amp)   # Q
    sig += _gaussian(t, beat_t,        0.008, 1.00 * amp)   # R
    sig -= _gaussian(t, beat_t + 0.02, 0.005, 0.03 * amp)   # S
    sig += _gaussian(t, beat_t + 0.20, 0.040, 0.20 * amp)   # T
    return sig


def _afib_beat(t, beat_t, amp=1.0):
    """Beat without P wave + irregular."""
    sig = np.zeros_like(t)
    sig += _gaussian(t, beat_t,        0.008, amp)
    sig -= _gaussian(t, beat_t + 0.02, 0.005, 0.03 * amp)
    sig += _gaussian(t, beat_t + 0.18, 0.040, 0.18 * amp)
    # fibrilatory baseline noise handled outside
    return sig


def _stemi_beat(t, beat_t, lead_idx, amp=1.0):
    """STEMI pattern — elevated ST in V1-V4 (leads 6-9)."""
    sig = _normal_beat(t, beat_t, amp)
    if lead_idx in (6, 7, 8, 9):  # V1-V4
        st_start = beat_t + 0.04
        st_end   = beat_t + 0.18
        mask = (t >= st_start) & (t <= st_end)
        sig[mask] += 0.3 * amp   # ST elevation
    return sig


def _lbbb_beat(t, beat_t, lead_idx, amp=1.0):
    """Left bundle branch block — wide QRS, notched R."""
    sig = np.zeros_like(t)
    sig += _gaussian(t, beat_t - 0.16, 0.020, 0.10 * amp)
    # Wide notched R
    sig += _gaussian(t, beat_t - 0.01, 0.015, 0.5 * amp)
    sig += _gaussian(t, beat_t + 0.04, 0.015, 0.5 * amp)
    if lead_idx in (0, 1, 4, 5, 9, 10, 11):  # Left leads
        sig += _gaussian(t, beat_t + 0.22, 0.045, 0.25 * amp)
    else:
        sig -= _gaussian(t, beat_t + 0.02, 0.010, 0.4 * amp)  # Deep S
        sig -= _gaussian(t, beat_t + 0.22, 0.035, 0.15 * amp)  # Inverted T
    return sig


def _vt_beat(t, beat_t, lead_idx, amp=1.0):
    """Ventricular tachycardia — wide, bizarre complex."""
    sig = np.zeros_like(t)
    phase = lead_idx * 0.15
    sig += _gaussian(t, beat_t + 0.00, 0.020, amp * np.cos(phase))
    sig += _gaussian(t, beat_t + 0.04, 0.015, -amp * 0.6 * np.sin(phase))
    sig += _gaussian(t, beat_t + 0.08, 0.025, amp * 0.4 * np.cos(phase + 1))
    sig += _gaussian(t, beat_t + 0.14, 0.040, amp * 0.3)
    return sig


def _lead_amplitude(lead_idx: int, ecg_type: str) -> float:
    """Per-lead amplitude scaling to mimic realistic 12-lead patterns."""
    base = [0.8, 1.2, 0.6, -1.0, 0.3, 1.0, -0.5, 0.8, 1.3, 1.4, 1.2, 0.9]
    amp = base[lead_idx % 12]
    if ecg_type == "stemi":
        boost = [0.8, 1.0, 0.8, -0.9, 0.5, 1.0, 1.4, 1.6, 1.5, 1.4, 1.0, 0.8]
        amp = boost[lead_idx % 12]
    return amp


def generate_ecg(ecg_type: str, hr: float = 72, noise: float = 0.025,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate (12, N) ECG array for the given morphology type."""
    if rng is None:
        rng = np.random.default_rng(42)

    t = _t_axis()
    period = 60.0 / hr
    ecg = np.zeros((N_CHANNELS, N), dtype=np.float32)

    for ch in range(N_CHANNELS):
        amp = _lead_amplitude(ch, ecg_type)

        if ecg_type == "normal":
            for bt in np.arange(0.2, DURATION, period):
                ecg[ch] += _normal_beat(t, bt, amp)

        elif ecg_type == "afib":
            # Irregular intervals
            beats = []
            cur = 0.2
            while cur < DURATION:
                beats.append(cur)
                jitter = rng.uniform(-0.15, 0.20) * period
                cur += max(0.3, period + jitter)
            for bt in beats:
                ecg[ch] += _afib_beat(t, bt, amp)
            # Fibrilatory baseline
            freqs = rng.uniform(4, 8, 6)
            phases = rng.uniform(0, 2 * np.pi, 6)
            for f, ph in zip(freqs, phases):
                ecg[ch] += 0.04 * np.sin(2 * np.pi * f * t + ph)

        elif ecg_type == "stemi":
            for bt in np.arange(0.2, DURATION, period):
                ecg[ch] += _stemi_beat(t, bt, ch, amp)

        elif ecg_type == "bradycardia":
            for bt in np.arange(0.4, DURATION, period):
                ecg[ch] += _normal_beat(t, bt, amp * 1.1, hr)

        elif ecg_type == "tachycardia":
            for bt in np.arange(0.1, DURATION, period):
                ecg[ch] += _normal_beat(t, bt, amp * 0.85, hr)

        elif ecg_type == "lbbb":
            for bt in np.arange(0.2, DURATION, period):
                ecg[ch] += _lbbb_beat(t, bt, ch, amp)

        elif ecg_type == "vt":
            for bt in np.arange(0.1, DURATION, period):
                ecg[ch] += _vt_beat(t, bt, ch, amp)

        elif ecg_type == "chf":
            # Low voltage, T-wave changes
            for bt in np.arange(0.2, DURATION, period):
                ecg[ch] += _normal_beat(t, bt, amp * 0.55, hr)
            # Inverted T in lateral leads
            if ch in (0, 4, 9, 10, 11):
                for bt in np.arange(0.2, DURATION, period):
                    ecg[ch] -= _gaussian(t, bt + 0.20, 0.040, 0.12 * abs(amp))

        ecg[ch] += rng.normal(0, noise, N).astype(np.float32)

    return ecg


def generate_tabular(
    ecg_type: str,
    vitals: Vital,
    labs: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a 1602-dim tabular feature vector matching the training data feature encoding.

    Features 0-9 are clinical proxies (same encoding as generate_demo_data.py):
      0: HR (normalised 0-1 over 30-200 bpm)
      1: troponin proxy  (0=normal, 1=very high)
      2: BNP proxy       (0=normal, 1=very high)
      3: lactate proxy   (0=normal, 1=very high)
      4: creatinine proxy
      5: glucose proxy
      6: WBC proxy
      7: platelet proxy  (1=normal/high, 0=thrombocytopenic)
      8: SBP proxy       (0=shock, 1=normal/hypertensive)
      9: SpO2 proxy      (0=severe hypoxia, 1=normal)
    Features 10-1601: standard-normal noise (matches training distribution).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if labs is None:
        labs = {}

    feat = rng.standard_normal(N_FEATURES).astype(np.float32)
    feat = np.clip(feat, -4.0, 4.0)

    def _n(val: float, lo: float, hi: float) -> float:
        return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))

    # Feature 0: normalised HR
    feat[0] = _n(vitals.hr, 30.0, 200.0) + float(rng.normal(0, 0.02))

    # Feature 1: troponin (ng/mL; >10 = very high)
    trop = labs.get("troponin_i", labs.get("troponin", 0.0)) or 0.0
    feat[1] = _n(trop, 0.0, 10.0) + float(rng.normal(0, 0.05))

    # Feature 2: BNP (pg/mL; >5000 = very high)
    bnp = labs.get("bnp", 0.0) or 0.0
    feat[2] = _n(bnp, 0.0, 5000.0) + float(rng.normal(0, 0.05))

    # Feature 3: lactate (mmol/L; normal <2, very high >8)
    lac = labs.get("lactate", 0.8) or 0.8
    feat[3] = _n(lac, 0.0, 8.0) + float(rng.normal(0, 0.05))

    # Feature 4: creatinine (mg/dL; normal ~1.0, high ~5.0)
    cr = labs.get("creatinine", 1.0) or 1.0
    feat[4] = _n(cr, 0.5, 5.0) + float(rng.normal(0, 0.04))

    # Feature 5: glucose (blood glucose normalised; normal ~100 mg/dL)
    glc = labs.get("glucose", 110.0) or 110.0
    feat[5] = _n(glc, 60.0, 400.0) + float(rng.normal(0, 0.04))

    # Feature 6: WBC (k/uL; normal 4-11, very high >25)
    wbc = labs.get("wbc", 8.0) or 8.0
    feat[6] = _n(wbc, 2.0, 25.0) + float(rng.normal(0, 0.05))

    # Feature 7: platelets (k/uL; normal 150-400, low <50)
    plt_ = labs.get("plt", labs.get("platelets", 250.0)) or 250.0
    feat[7] = _n(plt_, 20.0, 400.0) + float(rng.normal(0, 0.05))

    # Feature 8: SBP proxy (0=shock <70 mmHg, 1=normal/hypertensive)
    feat[8] = _n(vitals.sbp, 60.0, 160.0) + float(rng.normal(0, 0.04))

    # Feature 9: SpO2 proxy (0=severe hypoxia, 1=normal)
    feat[9] = _n(vitals.spo2, 75.0, 100.0) + float(rng.normal(0, 0.03))

    return np.clip(feat, -4.0, 4.0).astype(np.float32)


# ── Patient roster ─────────────────────────────────────────────────────────────

PATIENTS: List[PatientProfile] = [
    PatientProfile(
        patient_id="pt_001",
        name="James Morrison",
        age=67, gender="Male", room="ICU-3A",
        diagnosis="STEMI — Anterior Wall MI",
        history="Presented with 2h chest pain, diaphoresis. Cath lab pending. Troponin I: 8.4 ng/mL (↑↑). EF 35%.",
        vitals=Vital(hr=88, sbp=108, dbp=68, spo2=94, rr=20, temp=37.1, gcs=14, map_=81),
        labs={"troponin_i": 8.4, "ck_mb": 42, "bnp": 680, "creatinine": 1.3, "wbc": 12.1, "lactate": 2.8},
        ecg_summary="ST elevation V1-V4, reciprocal changes in II/III/aVF. Rate 88 bpm.",
        risk_narrative="Anterior STEMI with reduced EF. High risk for cardiogenic shock and cardiac arrest.",
        ecg_type="stemi",
        feature_bias={},
    ),
    PatientProfile(
        patient_id="pt_002",
        name="Sarah Chen",
        age=54, gender="Female", room="ICU-2B",
        diagnosis="Atrial Fibrillation with RVR + Sepsis",
        history="A-fib with rapid ventricular response. Known AF, non-compliant with warfarin. Blood cultures pending.",
        vitals=Vital(hr=138, sbp=92, dbp=54, spo2=96, rr=24, temp=38.9, gcs=13, map_=67),
        labs={"wbc": 18.4, "lactate": 3.6, "creatinine": 1.8, "inr": 1.2, "procalcitonin": 12.4, "bnp": 420},
        ecg_summary="Irregularly irregular, no P waves, rate 138 bpm. Tachycardic response.",
        risk_narrative="Septic shock physiology with AF-RVR. High risk for ICU transfer, vasopressor requirement, and mortality.",
        ecg_type="afib",
        feature_bias={},
    ),
    PatientProfile(
        patient_id="pt_003",
        name="Marcus Williams",
        age=72, gender="Male", room="ICU-1C",
        diagnosis="Ventricular Tachycardia — Hemodynamically Unstable",
        history="Found unresponsive. EMS initiated CPR × 3 min. ROSC achieved. Previous MI 2014. ICD in situ.",
        vitals=Vital(hr=156, sbp=74, dbp=42, spo2=88, rr=28, temp=36.3, gcs=8, map_=53),
        labs={"troponin_i": 15.2, "lactate": 6.8, "ph": 7.18, "k": 5.9, "creatinine": 2.4, "ck_mb": 88},
        ecg_summary="Wide complex tachycardia 156 bpm, left-axis deviation. V-tach morphology. ICD shocks × 2.",
        risk_narrative="Post-arrest cardiogenic shock. Extremely high mortality risk, cardiac arrest recurrence likely.",
        ecg_type="vt",
        feature_bias={},
    ),
    PatientProfile(
        patient_id="pt_004",
        name="Emily Rodriguez",
        age=45, gender="Female", room="ICU-4D",
        diagnosis="Acute Respiratory Failure — ARDS (COVID-19)",
        history="Day 5 ICU admission. Intubated day 3. FiO2 80%, PEEP 14. PaO2/FiO2 ratio 112.",
        vitals=Vital(hr=102, sbp=118, dbp=72, spo2=91, rr=22, temp=38.4, gcs=10, map_=87),
        labs={"wbc": 14.2, "crp": 184, "d_dimer": 4.2, "ferritin": 1820, "il6": 68, "lactate": 2.1},
        ecg_summary="Sinus tachycardia 102 bpm. Nonspecific ST changes. Right heart strain pattern.",
        risk_narrative="ARDS with high FiO2 requirement. Moderate mortality risk, high risk for prolonged ventilation.",
        ecg_type="tachycardia",
        feature_bias={},
    ),
    PatientProfile(
        patient_id="pt_005",
        name="Robert Kim",
        age=58, gender="Male", room="ICU-2A",
        diagnosis="Acute Decompensated Heart Failure",
        history="Known ischemic cardiomyopathy EF 25%. Admitted with 10kg weight gain, orthopnea, JVD. BNP 4800.",
        vitals=Vital(hr=58, sbp=88, dbp=58, spo2=90, rr=26, temp=36.8, gcs=15, map_=68),
        labs={"bnp": 4800, "creatinine": 2.2, "na": 128, "k": 5.1, "albumin": 2.8, "troponin_i": 0.9},
        ecg_summary="Bradycardia 58 bpm, LBBB morphology, low voltage. Q waves inferior leads.",
        risk_narrative="Decompensated CHF with cardiorenal syndrome. High risk for worsening AKI and vasopressor need.",
        ecg_type="chf",
        feature_bias={},
    ),
    PatientProfile(
        patient_id="pt_006",
        name="Diane Foster",
        age=63, gender="Female", room="ICU-3C",
        diagnosis="Septic Shock — Urosepsis",
        history="Presented febrile, hypotensive, altered. UTI source. E. coli bacteremia. On norepinephrine 0.18 mcg/kg/min.",
        vitals=Vital(hr=118, sbp=82, dbp=44, spo2=95, rr=22, temp=39.2, gcs=11, map_=57),
        labs={"wbc": 22.6, "lactate": 4.1, "procalcitonin": 28.4, "creatinine": 3.1, "bili": 2.8, "inr": 1.8},
        ecg_summary="Sinus tachycardia 118 bpm. Diffuse ST-T changes. QTc 480ms.",
        risk_narrative="Septic shock on vasopressors. High risk for multi-organ failure, AKI progression, and mortality.",
        ecg_type="tachycardia",
        feature_bias={},
    ),
    PatientProfile(
        patient_id="pt_007",
        name="Thomas Nguyen",
        age=78, gender="Male", room="ICU-1A",
        diagnosis="Complete Heart Block — Awaiting Pacemaker",
        history="Syncope × 2 this week. Found to have complete AV dissociation. Temporary pacing wire placed. Cardiomegaly.",
        vitals=Vital(hr=38, sbp=96, dbp=62, spo2=96, rr=16, temp=36.6, gcs=15, map_=73),
        labs={"k": 6.2, "digoxin": 2.8, "creatinine": 1.9, "ck_mb": 12, "tsh": 8.4, "ca": 8.1},
        ecg_summary="Complete heart block 38 bpm. P waves independent of QRS. Wide escape rhythm.",
        risk_narrative="Complete AV block with bradycardia. Risk of syncope, hemodynamic collapse if temporary pacing fails.",
        ecg_type="bradycardia",
        feature_bias={},
    ),
    PatientProfile(
        patient_id="pt_008",
        name="Priya Patel",
        age=41, gender="Female", room="ICU-4B",
        diagnosis="LBBB — New Onset, Anterior Ischemia",
        history="New LBBB on routine ECG post-knee surgery. Troponin trending up. Echo shows anterior wall motion abnormality.",
        vitals=Vital(hr=76, sbp=124, dbp=78, spo2=98, rr=14, temp=36.9, gcs=15, map_=93),
        labs={"troponin_i": 2.1, "ck_mb": 18, "ldl": 168, "hba1c": 7.8, "creatinine": 0.9, "bnp": 180},
        ecg_summary="LBBB morphology, QRS 148ms. Positive Sgarbossa criteria. Rate 76 bpm.",
        risk_narrative="New LBBB equivalent to STEMI. Moderate risk for cardiac arrest and in-hospital mortality.",
        ecg_type="lbbb",
        feature_bias={},
    ),
]

PATIENT_MAP = {p.patient_id: p for p in PATIENTS}


def get_patient_ecg_and_tabular(patient_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (ecg: (12, N), tabular: (1602,)) for a given patient."""
    profile = PATIENT_MAP[patient_id]
    rng = np.random.default_rng(int(patient_id.split("_")[1]))
    ecg = generate_ecg(profile.ecg_type, hr=profile.vitals.hr, rng=rng)
    tab = generate_tabular(profile.ecg_type, profile.vitals, labs=profile.labs, rng=rng)
    return ecg, tab
