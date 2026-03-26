"""
Generate clinically-correlated demo data for MDS-ICU training.

Each synthetic patient is assigned one of 8 clinical archetypes (normal, MI,
sepsis, CHF, ARDS, AFib, tachycardia, bradycardia).  ECG morphology and
tabular features are biased to match the archetype, and outcome labels are
assigned with probabilities derived from published ICU epidemiology.

This gives the model real signal to learn from so inference produces
clinically meaningful, patient-specific predictions (AUROC >> 0.5).

Creates:
  data/memmap/memmap.npy          - ECG memmap (N * seq_len, 12)
  data/memmap/memmap_meta.npz     - metadata
  data/memmap/df_memmap.pkl       - DataFrame (data, label, features, strat_fold)
  data/memmap/mds_icu_preprocessed2_ecg.parquet  - same as df
  data/memmap/lbl_itos.pkl        - list of 33 label names
  data/memmap/mean.npy            - per-channel mean (zeros)
  data/memmap/std.npy             - per-channel std (ones)

Run from backend/src/:
  python generate_demo_data.py
"""

import numpy as np
import pandas as pd
import pickle
import pathlib

# ── Parameters ─────────────────────────────────────────────────────────────────
N_SAMPLES   = 1200      # total synthetic patients (more = better model)
SEQ_LEN     = 1000      # timesteps per sample (10 s at 100 Hz)
N_CHANNELS  = 12        # ECG leads
N_FEATURES  = 1602      # tabular features
N_LABELS    = 33        # clinical outcome labels
N_FOLDS     = 10        # stratification folds (1..10)
FS          = 100.0     # sampling frequency Hz
DTYPE       = np.float32
SEED        = 42
# ──────────────────────────────────────────────────────────────────────────────

LABEL_NAMES = [
    # Mortality (2)
    "mortality_inhospital", "mortality_30d",
    # Medication (8)
    "med_vasopressor", "med_antibiotic", "med_sedative",
    "med_anticoagulant", "med_diuretic", "med_insulin",
    "med_steroid", "med_antiarrhythmic",
    # Deterioration (9)
    "detr_icu_transfer", "detr_intubation", "detr_cpr",
    "detr_renal_replacement", "detr_shock", "detr_sepsis",
    "detr_respiratory_failure", "detr_cardiac_arrest", "detr_stroke",
    # Organ Dysfunction (14)
    "od_aki", "od_liver_failure", "od_coagulopathy",
    "od_thrombocytopenia", "od_anemia", "od_electrolyte_imbalance",
    "od_hyperglycemia", "od_hypoglycemia", "od_hypotension",
    "od_hypertension", "od_arrhythmia", "od_heart_failure",
    "od_pneumonia", "od_ards",
]
assert len(LABEL_NAMES) == N_LABELS

LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

# ── Clinical archetype definitions ─────────────────────────────────────────────
# Each archetype has: hr_range, label probabilities, ECG morphology params,
# and tabular feature biases (scaled 0-1 on clinical-proxy features).

# fmt: off
ARCHETYPES = {
    "normal": {
        "weight": 0.22,
        "hr": (60, 85),
        "ecg": dict(r_amp=1.0, noise=0.04, st_elev=0.0, qrs_wide=False, irreg=False),
        "labs": dict(troponin=0.0, bnp=0.0, lactate=0.0, creatinine=0.3, glucose=0.45, wbc=0.3, plt=0.6),
        "labels": {
            "mortality_inhospital": 0.02, "mortality_30d": 0.03,
            "med_vasopressor": 0.02, "med_antibiotic": 0.06, "med_sedative": 0.03,
            "med_anticoagulant": 0.08, "med_diuretic": 0.10, "med_insulin": 0.12,
            "med_steroid": 0.04, "med_antiarrhythmic": 0.04,
            "detr_icu_transfer": 0.04, "detr_intubation": 0.02, "detr_cpr": 0.01,
            "detr_renal_replacement": 0.02, "detr_shock": 0.02, "detr_sepsis": 0.03,
            "detr_respiratory_failure": 0.03, "detr_cardiac_arrest": 0.01, "detr_stroke": 0.02,
            "od_aki": 0.05, "od_liver_failure": 0.02, "od_coagulopathy": 0.03,
            "od_thrombocytopenia": 0.04, "od_anemia": 0.10, "od_electrolyte_imbalance": 0.12,
            "od_hyperglycemia": 0.15, "od_hypoglycemia": 0.03, "od_hypotension": 0.04,
            "od_hypertension": 0.18, "od_arrhythmia": 0.05, "od_heart_failure": 0.05,
            "od_pneumonia": 0.06, "od_ards": 0.02,
        },
    },
    "mi": {  # Myocardial infarction / STEMI
        "weight": 0.12,
        "hr": (75, 115),
        "ecg": dict(r_amp=0.7, noise=0.08, st_elev=0.25, qrs_wide=False, irreg=False),
        "labs": dict(troponin=0.92, bnp=0.55, lactate=0.45, creatinine=0.45, glucose=0.60, wbc=0.55, plt=0.45),
        "labels": {
            "mortality_inhospital": 0.35, "mortality_30d": 0.42,
            "med_vasopressor": 0.55, "med_antibiotic": 0.25, "med_sedative": 0.30,
            "med_anticoagulant": 0.80, "med_diuretic": 0.35, "med_insulin": 0.20,
            "med_steroid": 0.10, "med_antiarrhythmic": 0.45,
            "detr_icu_transfer": 0.75, "detr_intubation": 0.25, "detr_cpr": 0.30,
            "detr_renal_replacement": 0.20, "detr_shock": 0.50, "detr_sepsis": 0.12,
            "detr_respiratory_failure": 0.22, "detr_cardiac_arrest": 0.55, "detr_stroke": 0.08,
            "od_aki": 0.35, "od_liver_failure": 0.10, "od_coagulopathy": 0.30,
            "od_thrombocytopenia": 0.15, "od_anemia": 0.25, "od_electrolyte_imbalance": 0.40,
            "od_hyperglycemia": 0.30, "od_hypoglycemia": 0.05, "od_hypotension": 0.45,
            "od_hypertension": 0.20, "od_arrhythmia": 0.55, "od_heart_failure": 0.65,
            "od_pneumonia": 0.12, "od_ards": 0.15,
        },
    },
    "sepsis": {  # Septic shock / severe infection
        "weight": 0.14,
        "hr": (105, 145),
        "ecg": dict(r_amp=0.8, noise=0.10, st_elev=0.0, qrs_wide=False, irreg=False),
        "labs": dict(troponin=0.30, bnp=0.35, lactate=0.85, creatinine=0.70, glucose=0.55, wbc=0.88, plt=0.25),
        "labels": {
            "mortality_inhospital": 0.40, "mortality_30d": 0.48,
            "med_vasopressor": 0.82, "med_antibiotic": 0.95, "med_sedative": 0.40,
            "med_anticoagulant": 0.30, "med_diuretic": 0.20, "med_insulin": 0.30,
            "med_steroid": 0.50, "med_antiarrhythmic": 0.12,
            "detr_icu_transfer": 0.88, "detr_intubation": 0.45, "detr_cpr": 0.15,
            "detr_renal_replacement": 0.35, "detr_shock": 0.78, "detr_sepsis": 0.95,
            "detr_respiratory_failure": 0.50, "detr_cardiac_arrest": 0.18, "detr_stroke": 0.05,
            "od_aki": 0.70, "od_liver_failure": 0.35, "od_coagulopathy": 0.55,
            "od_thrombocytopenia": 0.50, "od_anemia": 0.40, "od_electrolyte_imbalance": 0.60,
            "od_hyperglycemia": 0.50, "od_hypoglycemia": 0.10, "od_hypotension": 0.80,
            "od_hypertension": 0.05, "od_arrhythmia": 0.30, "od_heart_failure": 0.25,
            "od_pneumonia": 0.45, "od_ards": 0.30,
        },
    },
    "chf": {  # Congestive heart failure / decompensated
        "weight": 0.12,
        "hr": (52, 72),
        "ecg": dict(r_amp=0.5, noise=0.06, st_elev=-0.1, qrs_wide=True, irreg=False),
        "labs": dict(troponin=0.25, bnp=0.95, lactate=0.30, creatinine=0.60, glucose=0.40, wbc=0.35, plt=0.45),
        "labels": {
            "mortality_inhospital": 0.22, "mortality_30d": 0.30,
            "med_vasopressor": 0.20, "med_antibiotic": 0.15, "med_sedative": 0.12,
            "med_anticoagulant": 0.50, "med_diuretic": 0.92, "med_insulin": 0.20,
            "med_steroid": 0.08, "med_antiarrhythmic": 0.35,
            "detr_icu_transfer": 0.60, "detr_intubation": 0.22, "detr_cpr": 0.12,
            "detr_renal_replacement": 0.25, "detr_shock": 0.22, "detr_sepsis": 0.08,
            "detr_respiratory_failure": 0.35, "detr_cardiac_arrest": 0.15, "detr_stroke": 0.08,
            "od_aki": 0.55, "od_liver_failure": 0.20, "od_coagulopathy": 0.15,
            "od_thrombocytopenia": 0.15, "od_anemia": 0.35, "od_electrolyte_imbalance": 0.55,
            "od_hyperglycemia": 0.20, "od_hypoglycemia": 0.04, "od_hypotension": 0.20,
            "od_hypertension": 0.30, "od_arrhythmia": 0.55, "od_heart_failure": 0.92,
            "od_pneumonia": 0.15, "od_ards": 0.12,
        },
    },
    "ards": {  # ARDS / severe respiratory failure
        "weight": 0.10,
        "hr": (100, 130),
        "ecg": dict(r_amp=0.9, noise=0.12, st_elev=0.0, qrs_wide=False, irreg=False),
        "labs": dict(troponin=0.20, bnp=0.30, lactate=0.60, creatinine=0.55, glucose=0.50, wbc=0.70, plt=0.30),
        "labels": {
            "mortality_inhospital": 0.45, "mortality_30d": 0.52,
            "med_vasopressor": 0.55, "med_antibiotic": 0.70, "med_sedative": 0.85,
            "med_anticoagulant": 0.35, "med_diuretic": 0.30, "med_insulin": 0.30,
            "med_steroid": 0.60, "med_antiarrhythmic": 0.10,
            "detr_icu_transfer": 0.90, "detr_intubation": 0.90, "detr_cpr": 0.10,
            "detr_renal_replacement": 0.30, "detr_shock": 0.45, "detr_sepsis": 0.55,
            "detr_respiratory_failure": 0.95, "detr_cardiac_arrest": 0.12, "detr_stroke": 0.04,
            "od_aki": 0.45, "od_liver_failure": 0.25, "od_coagulopathy": 0.40,
            "od_thrombocytopenia": 0.35, "od_anemia": 0.40, "od_electrolyte_imbalance": 0.50,
            "od_hyperglycemia": 0.35, "od_hypoglycemia": 0.08, "od_hypotension": 0.45,
            "od_hypertension": 0.08, "od_arrhythmia": 0.25, "od_heart_failure": 0.25,
            "od_pneumonia": 0.60, "od_ards": 0.95,
        },
    },
    "afib": {  # Atrial fibrillation
        "weight": 0.10,
        "hr": (90, 145),
        "ecg": dict(r_amp=1.0, noise=0.08, st_elev=0.0, qrs_wide=False, irreg=True),
        "labs": dict(troponin=0.12, bnp=0.40, lactate=0.20, creatinine=0.40, glucose=0.42, wbc=0.35, plt=0.55),
        "labels": {
            "mortality_inhospital": 0.12, "mortality_30d": 0.18,
            "med_vasopressor": 0.10, "med_antibiotic": 0.15, "med_sedative": 0.08,
            "med_anticoagulant": 0.85, "med_diuretic": 0.30, "med_insulin": 0.15,
            "med_steroid": 0.05, "med_antiarrhythmic": 0.88,
            "detr_icu_transfer": 0.35, "detr_intubation": 0.08, "detr_cpr": 0.05,
            "detr_renal_replacement": 0.08, "detr_shock": 0.10, "detr_sepsis": 0.08,
            "detr_respiratory_failure": 0.10, "detr_cardiac_arrest": 0.10, "detr_stroke": 0.35,
            "od_aki": 0.15, "od_liver_failure": 0.05, "od_coagulopathy": 0.15,
            "od_thrombocytopenia": 0.08, "od_anemia": 0.20, "od_electrolyte_imbalance": 0.30,
            "od_hyperglycemia": 0.15, "od_hypoglycemia": 0.03, "od_hypotension": 0.12,
            "od_hypertension": 0.35, "od_arrhythmia": 0.92, "od_heart_failure": 0.35,
            "od_pneumonia": 0.08, "od_ards": 0.04,
        },
    },
    "vt": {  # Ventricular tachycardia / post-arrest
        "weight": 0.10,
        "hr": (150, 195),
        "ecg": dict(r_amp=1.3, noise=0.15, st_elev=0.05, qrs_wide=True, irreg=True),
        "labs": dict(troponin=0.70, bnp=0.45, lactate=0.75, creatinine=0.55, glucose=0.55, wbc=0.50, plt=0.35),
        "labels": {
            "mortality_inhospital": 0.55, "mortality_30d": 0.62,
            "med_vasopressor": 0.70, "med_antibiotic": 0.30, "med_sedative": 0.50,
            "med_anticoagulant": 0.40, "med_diuretic": 0.20, "med_insulin": 0.20,
            "med_steroid": 0.15, "med_antiarrhythmic": 0.90,
            "detr_icu_transfer": 0.92, "detr_intubation": 0.55, "detr_cpr": 0.70,
            "detr_renal_replacement": 0.30, "detr_shock": 0.65, "detr_sepsis": 0.15,
            "detr_respiratory_failure": 0.40, "detr_cardiac_arrest": 0.85, "detr_stroke": 0.06,
            "od_aki": 0.45, "od_liver_failure": 0.25, "od_coagulopathy": 0.35,
            "od_thrombocytopenia": 0.25, "od_anemia": 0.30, "od_electrolyte_imbalance": 0.55,
            "od_hyperglycemia": 0.25, "od_hypoglycemia": 0.05, "od_hypotension": 0.55,
            "od_hypertension": 0.08, "od_arrhythmia": 0.90, "od_heart_failure": 0.55,
            "od_pneumonia": 0.15, "od_ards": 0.20,
        },
    },
    "bradycardia": {  # Complete heart block / severe bradycardia
        "weight": 0.10,
        "hr": (30, 50),
        "ecg": dict(r_amp=0.9, noise=0.05, st_elev=0.0, qrs_wide=True, irreg=False),
        "labs": dict(troponin=0.15, bnp=0.30, lactate=0.30, creatinine=0.45, glucose=0.42, wbc=0.32, plt=0.55),
        "labels": {
            "mortality_inhospital": 0.18, "mortality_30d": 0.25,
            "med_vasopressor": 0.22, "med_antibiotic": 0.10, "med_sedative": 0.12,
            "med_anticoagulant": 0.25, "med_diuretic": 0.20, "med_insulin": 0.15,
            "med_steroid": 0.08, "med_antiarrhythmic": 0.50,
            "detr_icu_transfer": 0.60, "detr_intubation": 0.10, "detr_cpr": 0.25,
            "detr_renal_replacement": 0.15, "detr_shock": 0.20, "detr_sepsis": 0.06,
            "detr_respiratory_failure": 0.10, "detr_cardiac_arrest": 0.40, "detr_stroke": 0.06,
            "od_aki": 0.20, "od_liver_failure": 0.08, "od_coagulopathy": 0.10,
            "od_thrombocytopenia": 0.10, "od_anemia": 0.18, "od_electrolyte_imbalance": 0.35,
            "od_hyperglycemia": 0.12, "od_hypoglycemia": 0.05, "od_hypotension": 0.30,
            "od_hypertension": 0.15, "od_arrhythmia": 0.80, "od_heart_failure": 0.35,
            "od_pneumonia": 0.08, "od_ards": 0.04,
        },
    },
}
# fmt: on

ARCHETYPE_NAMES = list(ARCHETYPES.keys())
ARCHETYPE_WEIGHTS = np.array([ARCHETYPES[k]["weight"] for k in ARCHETYPE_NAMES])
ARCHETYPE_WEIGHTS /= ARCHETYPE_WEIGHTS.sum()


# ── Helpers ────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)


def _pqrst_beat(
    fs: float,
    hr: float = 75.0,
    noise_sigma: float = 0.05,
    st_elev: float = 0.0,
    r_amp: float = 1.0,
    qrs_wide: bool = False,
    rr_jitter: float = 0.0,
) -> np.ndarray:
    """Produce one synthetic PQRST ECG beat for Lead II."""
    beat_len = max(20, int(fs * 60 / hr * (1.0 + rng.uniform(-rr_jitter, rr_jitter))))
    t = np.linspace(0, 1, beat_len)
    qrs_sigma = 0.006 if not qrs_wide else 0.012
    # P wave
    p = 0.15 * np.exp(-((t - 0.20) ** 2) / (2 * 0.007 ** 2))
    # QRS
    q = -0.10 * np.exp(-((t - 0.35) ** 2) / (2 * qrs_sigma ** 2))
    r =  r_amp * np.exp(-((t - 0.38) ** 2) / (2 * (qrs_sigma * 0.67) ** 2))
    s = -0.20 * np.exp(-((t - 0.41) ** 2) / (2 * qrs_sigma ** 2))
    # T wave + ST elevation/depression
    tw = 0.35 * np.exp(-((t - 0.60) ** 2) / (2 * 0.012 ** 2))
    st = st_elev * np.exp(-((t - 0.50) ** 2) / (2 * 0.025 ** 2))
    beat = p + q + r + s + tw + st + noise_sigma * rng.standard_normal(beat_len)
    return beat.astype(DTYPE)


def _lead_scale(lead_idx: int) -> float:
    scales = [1.0, 1.4, 0.5, -1.0, 0.3, 0.7,
              0.2, 0.6, 1.1, 1.3, 1.2, 0.9]
    return scales[lead_idx]


def generate_ecg_for_archetype(seq_len: int, fs: float, archetype: str) -> np.ndarray:
    """Generate a 12-lead ECG for a given archetype."""
    params = ARCHETYPES[archetype]["ecg"]
    hr = rng.uniform(*ARCHETYPES[archetype]["hr"])
    rr_jitter = 0.12 if params["irreg"] else 0.0
    beat = _pqrst_beat(
        fs, hr,
        noise_sigma=params["noise"],
        st_elev=params["st_elev"],
        r_amp=params["r_amp"],
        qrs_wide=params["qrs_wide"],
        rr_jitter=rr_jitter,
    )
    n_repeats = seq_len // len(beat) + 2
    signal = np.tile(beat, n_repeats)[:seq_len]
    ecg = np.stack([signal * _lead_scale(i) for i in range(N_CHANNELS)], axis=1)
    return ecg.astype(DTYPE), hr


def generate_tabular_for_archetype(archetype: str, hr: float) -> np.ndarray:
    """
    Generate 1602 tabular features.
    First 10 are clinically-motivated proxies correlated with the labels.
    Remaining 1592 are standard-normal noise (realistic for unmeasured features).
    """
    labs = ARCHETYPES[archetype]["labs"]
    features = rng.standard_normal(N_FEATURES).astype(DTYPE)
    # Feature 0: normalized HR (0-1 over 30-200 bpm range)
    features[0] = float((hr - 30.0) / 170.0 + rng.normal(0, 0.03))
    # Feature 1: troponin proxy
    features[1] = float(labs["troponin"] + rng.normal(0, 0.08))
    # Feature 2: BNP/pro-BNP proxy
    features[2] = float(labs["bnp"] + rng.normal(0, 0.08))
    # Feature 3: lactate proxy
    features[3] = float(labs["lactate"] + rng.normal(0, 0.08))
    # Feature 4: creatinine proxy
    features[4] = float(labs["creatinine"] + rng.normal(0, 0.06))
    # Feature 5: glucose proxy
    features[5] = float(labs["glucose"] + rng.normal(0, 0.06))
    # Feature 6: WBC proxy
    features[6] = float(labs["wbc"] + rng.normal(0, 0.07))
    # Feature 7: platelet proxy (inverted — high plt = healthy)
    features[7] = float(labs["plt"] + rng.normal(0, 0.06))
    # Feature 8: systolic BP proxy (low = shock)
    sbp_base = 0.75 - labs["lactate"] * 0.5
    features[8] = float(np.clip(sbp_base + rng.normal(0, 0.08), 0.0, 1.0))
    # Feature 9: SpO2 proxy (low = respiratory failure/ARDS)
    spo2_base = 0.95 - labs["lactate"] * 0.3
    features[9] = float(np.clip(spo2_base + rng.normal(0, 0.04), 0.0, 1.0))
    return np.clip(features, -4.0, 4.0)


def main():
    out = pathlib.Path("data/memmap")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating {N_SAMPLES} clinically-correlated samples ...")

    # ── Assign archetypes ─────────────────────────────────────────────────────
    archetype_indices = rng.choice(len(ARCHETYPE_NAMES), size=N_SAMPLES, p=ARCHETYPE_WEIGHTS)
    sample_archetypes = [ARCHETYPE_NAMES[i] for i in archetype_indices]

    # Print archetype distribution
    from collections import Counter
    counts = Counter(sample_archetypes)
    print("  Archetype distribution:")
    for name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {name:15s}: {cnt:4d} ({100*cnt/N_SAMPLES:.1f}%)")

    # ── Build memmap ──────────────────────────────────────────────────────────
    total_len = N_SAMPLES * SEQ_LEN
    memmap_path = out / "memmap.npy"
    memmap = np.memmap(memmap_path, dtype=DTYPE, mode="w+", shape=(total_len, N_CHANNELS))

    starts    = np.arange(N_SAMPLES, dtype=np.int64) * SEQ_LEN
    lengths   = np.full(N_SAMPLES, SEQ_LEN, dtype=np.int64)
    file_idxs = np.zeros(N_SAMPLES, dtype=np.int64)

    heart_rates  = np.zeros(N_SAMPLES, dtype=np.float32)
    all_features = np.zeros((N_SAMPLES, N_FEATURES), dtype=DTYPE)
    all_labels   = np.zeros((N_SAMPLES, N_LABELS),   dtype=DTYPE)

    for i in range(N_SAMPLES):
        arch = sample_archetypes[i]
        ecg, hr = generate_ecg_for_archetype(SEQ_LEN, FS, arch)
        memmap[starts[i]: starts[i] + SEQ_LEN] = ecg
        heart_rates[i] = hr

        # Tabular features
        all_features[i] = generate_tabular_for_archetype(arch, hr)

        # Labels — sampled from archetype-specific probabilities + small noise
        lp = ARCHETYPES[arch]["labels"]
        probs = np.array([lp[l] for l in LABEL_NAMES], dtype=np.float64)
        # Add small individual variation
        probs = np.clip(probs + rng.normal(0, 0.05, size=N_LABELS), 0.02, 0.98)
        all_labels[i] = (rng.random(N_LABELS) < probs).astype(DTYPE)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{N_SAMPLES} ({arch})")

    memmap.flush()
    del memmap
    print("memmap.npy written.")

    # ── Save memmap meta ──────────────────────────────────────────────────────
    meta = {"version": "1.0", "fs": FS, "channel_itos": LEAD_NAMES}
    np.savez(
        out / "memmap_meta.npz",
        start=starts,
        length=lengths,
        shape=np.array([[total_len, N_CHANNELS]], dtype=np.int64),
        file_idx=file_idxs,
        dtype=np.dtype(DTYPE).str,
        filenames=["memmap.npy"],
        meta=meta,
    )
    print("memmap_meta.npz written.")

    # ── Build DataFrame ───────────────────────────────────────────────────────
    folds = (np.arange(N_SAMPLES) % N_FOLDS) + 1
    df = pd.DataFrame({
        "data":       np.arange(N_SAMPLES, dtype=np.int64),
        "label":      list(all_labels),
        "features":   list(all_features),
        "strat_fold": folds,
        "archetype":  sample_archetypes,
    })

    df.to_pickle(out / "df_memmap.pkl", protocol=4)
    print("df_memmap.pkl written.")

    df.to_parquet(out / "mds_icu_preprocessed2_ecg.parquet")
    print("mds_icu_preprocessed2_ecg.parquet written.")

    # ── Label names ───────────────────────────────────────────────────────────
    with open(out / "lbl_itos.pkl", "wb") as f:
        pickle.dump(LABEL_NAMES, f, protocol=4)
    print("lbl_itos.pkl written.")

    # ── Normalisation stats ───────────────────────────────────────────────────
    np.save(out / "mean.npy", np.zeros(N_CHANNELS, dtype=DTYPE))
    np.save(out / "std.npy",  np.ones(N_CHANNELS,  dtype=DTYPE))
    print("mean.npy / std.npy written.")

    # ── Summary ───────────────────────────────────────────────────────────────
    pos_rate = all_labels.mean(axis=0)
    print(f"\nDone. Dataset at {out.resolve()}")
    print(f"  Samples   : {N_SAMPLES}")
    print(f"  Shape     : ({SEQ_LEN} timesteps, {N_CHANNELS} channels)")
    print(f"  Features  : {N_FEATURES}")
    print(f"  Labels    : {N_LABELS} (avg positive rate: {pos_rate.mean():.1%})")
    print(f"  Folds     : {N_FOLDS} (1..{N_FOLDS})")
    print("\nLabel positive rates:")
    for name, rate in zip(LABEL_NAMES, pos_rate):
        bar = "█" * int(rate * 20)
        print(f"  {name:35s}: {rate:.2f}  {bar}")


if __name__ == "__main__":
    main()
