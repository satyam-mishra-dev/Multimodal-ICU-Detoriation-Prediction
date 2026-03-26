import yaml
from pathlib import Path
from fastapi import APIRouter, HTTPException

from ..schemas import FullConfigResponse, ConfigOption, TrainingConfig

router = APIRouter(prefix="/config", tags=["Configuration"])

CONF_DIR = Path(__file__).parent.parent.parent / "src" / "conf"
MAIN_CONFIG = CONF_DIR / "config_supervised_multimodal_mdsicu.yaml"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


@router.get("/current", response_model=FullConfigResponse)
def get_current_config():
    """Get the current training configuration (YAML + parsed)."""
    yaml_content = MAIN_CONFIG.read_text() if MAIN_CONFIG.exists() else "# Config file not found"
    parsed = _load_yaml(MAIN_CONFIG)

    options = [
        ConfigOption(key="predictor", label="Time Series Predictor", options=["s4", "none"], current="s4"),
        ConfigOption(key="encoder", label="Time Series Encoder", options=["none", "rnn"], current="none"),
        ConfigOption(key="fusion", label="Multimodal Fusion", options=["poolconcat", "concat", "tensor", "attention"], current="poolconcat"),
        ConfigOption(key="loss", label="Loss Function", options=["bcef", "bce", "ce", "cef", "mse"], current="bcef"),
        ConfigOption(key="metric", label="Evaluation Metric", options=["aurocagg", "auroc", "none"], current="aurocagg"),
        ConfigOption(key="static_encoder", label="Static/Tabular Encoder", options=["realmlp", "mlp", "none"], current="realmlp"),
    ]

    return FullConfigResponse(
        yaml_content=yaml_content,
        parsed=parsed,
        options=options,
    )


@router.get("/options")
def get_config_options():
    """Get all available configuration options for each component."""
    return {
        "predictor": {
            "label": "Time Series Predictor",
            "options": [
                {"value": "s4", "label": "S4 (Structured State Space)", "description": "State-of-the-art sequence model for long-range dependencies"},
                {"value": "none", "label": "None (Identity)", "description": "No predictor, use raw encoder output"},
            ],
        },
        "encoder": {
            "label": "Time Series Encoder",
            "options": [
                {"value": "none", "label": "None (Identity)", "description": "Pass raw signal to predictor"},
                {"value": "rnn", "label": "RNN / CNN Encoder", "description": "1D CNN for signal encoding"},
            ],
        },
        "fusion": {
            "label": "Multimodal Fusion Strategy",
            "options": [
                {"value": "poolconcat", "label": "Pool + Concatenate", "description": "Pool time series, concatenate with tabular features"},
                {"value": "concat", "label": "Concatenate", "description": "Direct concatenation of all features"},
                {"value": "tensor", "label": "Tensor Fusion", "description": "Zadeh et al. outer-product fusion"},
                {"value": "attention", "label": "Attention Fusion", "description": "Multi-head attention across modalities"},
            ],
        },
        "loss": {
            "label": "Loss Function",
            "options": [
                {"value": "bcef", "label": "Focal BCE (recommended)", "description": "Binary cross-entropy with focal modulation (γ=2) for imbalanced labels"},
                {"value": "bce", "label": "Binary Cross-Entropy", "description": "Standard BCE for multi-label classification"},
                {"value": "ce", "label": "Cross-Entropy", "description": "Multi-class cross-entropy"},
                {"value": "cef", "label": "Focal Cross-Entropy", "description": "Focal variant of multi-class CE"},
                {"value": "mse", "label": "MSE", "description": "Mean squared error for regression tasks"},
            ],
        },
        "optimizer": {
            "label": "Optimizer",
            "options": [
                {"value": "adam", "label": "Adam", "description": "Adaptive moment estimation optimizer"},
                {"value": "sgd", "label": "SGD", "description": "Stochastic gradient descent with momentum"},
            ],
        },
        "lr_schedule": {
            "label": "LR Schedule",
            "options": [
                {"value": "const", "label": "Constant", "description": "Fixed learning rate"},
                {"value": "warmup-cos", "label": "Warmup + Cosine Decay", "description": "Linear warmup then cosine annealing"},
                {"value": "warmup-poly", "label": "Warmup + Polynomial Decay", "description": "Linear warmup then polynomial decay"},
                {"value": "const-plateau", "label": "Reduce on Plateau", "description": "Reduce LR when metric stops improving"},
            ],
        },
    }


@router.get("/presets")
def get_config_presets():
    """Get predefined configuration presets."""
    return {
        "presets": [
            {
                "id": "default",
                "name": "MDS-ICU Default",
                "description": "The full paper configuration: S4 predictor + RealMLP + BCEFocal loss",
                "config": TrainingConfig().model_dump(),
            },
            {
                "id": "fast_debug",
                "name": "Fast Debug",
                "description": "Smaller model for quick debugging (2 epochs, small batch)",
                "config": TrainingConfig(epochs=2, batch_size=8, model_dim=64, s4_layers=2).model_dump(),
            },
            {
                "id": "eval_only",
                "name": "Evaluation Only",
                "description": "Skip training, only evaluate on test set using a saved checkpoint",
                "config": TrainingConfig(epochs=1, eval_only="best_model.ckpt").model_dump(),
            },
        ]
    }


@router.get("/architecture")
def get_architecture_info():
    """Get a description of the model architecture components."""
    return {
        "pipeline": [
            {
                "step": 1,
                "name": "ECG Input",
                "description": "12-lead ECG waveform",
                "details": "12 channels × 250 samples @ 100Hz (2.5 seconds per window)",
            },
            {
                "step": 2,
                "name": "Time Series Encoder",
                "description": "Optional signal encoding/tokenization",
                "details": "Default: Identity (NoEncoder). Alternative: 1D CNN RNN Encoder",
            },
            {
                "step": 3,
                "name": "S4 Predictor",
                "description": "Structured State Space sequence model",
                "details": "d_model=512, state_dim=8, 4 layers, bidirectional. Captures long-range temporal dependencies efficiently.",
            },
            {
                "step": 4,
                "name": "Pooling Head",
                "description": "Aggregate sequence representation",
                "details": "Global average/max pooling over time axis → fixed-size vector",
            },
            {
                "step": 5,
                "name": "RealMLP (Tabular)",
                "description": "Encode 1602 clinical tabular features",
                "details": "Quantile normalization → 4-layer MLP with SELU → 64-dim embedding. Handles both continuous and categorical features.",
            },
            {
                "step": 6,
                "name": "Fusion Head",
                "description": "Combine ECG and tabular representations",
                "details": "PoolingConcatFusion: concatenate pooled ECG + tabular → linear projection",
            },
            {
                "step": 7,
                "name": "Classification Head",
                "description": "Output predictions",
                "details": "Linear layer → 33 output logits → sigmoid → probabilities",
            },
            {
                "step": 8,
                "name": "BCEFocal Loss",
                "description": "Training objective",
                "details": "Binary cross-entropy with focal modulation γ=2. NaN-aware for missing labels.",
            },
        ],
        "label_groups": {
            "Mortality": ["in_hospital_mortality", "30_day_mortality", "90_day_mortality"],
            "Medication": ["vasopressor_use", "antibiotic_use", "anticoagulation", "diuretic_use",
                          "sedation_use", "insulin_use", "steroid_use", "blood_transfusion"],
            "Deterioration": ["mechanical_ventilation", "renal_replacement_therapy", "icu_readmission",
                             "cardiac_arrest", "septic_shock", "respiratory_failure",
                             "hemodynamic_instability", "new_arrhythmia", "pulmonary_embolism", "stroke"],
            "Organ Dysfunction": ["acute_kidney_injury", "acute_liver_failure", "coagulopathy",
                                  "thrombocytopenia", "hyponatremia", "hyperkalemia",
                                  "metabolic_acidosis", "respiratory_acidosis",
                                  "cardiac_dysfunction", "neurological_deterioration",
                                  "gastrointestinal_bleeding", "adrenal_insufficiency"],
        },
    }
