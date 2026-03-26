from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# ── Enums ────────────────────────────────────────────────────────────────────

class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class PredictorType(str, Enum):
    S4 = "s4"
    NONE = "none"


class EncoderType(str, Enum):
    NONE = "none"
    RNN = "rnn"


class FusionType(str, Enum):
    POOLCONCAT = "poolconcat"
    CONCAT = "concat"
    TENSOR = "tensor"
    ATTENTION = "attention"


class LossType(str, Enum):
    BCEF = "bcef"
    BCE = "bce"
    CE = "ce"
    CEF = "cef"
    MSE = "mse"


# ── Training ─────────────────────────────────────────────────────────────────

class TrainingConfig(BaseModel):
    epochs: int = Field(default=20, ge=1, le=1000)
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=1e-3, gt=0, le=1.0)
    weight_decay: float = Field(default=1e-3, ge=0)
    gpus: int = Field(default=0, ge=0)
    precision: int = Field(default=32)
    predictor: PredictorType = PredictorType.S4
    encoder: EncoderType = EncoderType.NONE
    fusion: FusionType = FusionType.POOLCONCAT
    loss: LossType = LossType.BCEF
    model_dim: int = Field(default=512)
    state_dim: int = Field(default=8)
    s4_layers: int = Field(default=4)
    input_channels: int = Field(default=12)
    input_channels_cont: int = Field(default=1602)
    fs: float = Field(default=100.0)
    input_size: int = Field(default=250)
    resume: str = Field(default="")
    eval_only: str = Field(default="")
    metadata: str = Field(default="run")


class TrainingStartResponse(BaseModel):
    run_id: str
    status: TrainingStatus
    message: str


class TrainingRun(BaseModel):
    run_id: str
    status: TrainingStatus
    config: TrainingConfig
    created_at: str
    updated_at: str
    pid: Optional[int] = None
    logs: List[str] = []
    metrics: Dict[str, Any] = {}


class TrainingRunSummary(BaseModel):
    run_id: str
    status: TrainingStatus
    created_at: str
    updated_at: str
    epochs_done: int = 0
    best_auroc: Optional[float] = None


# ── Inference ─────────────────────────────────────────────────────────────────

class PredictionLabel(BaseModel):
    label: str
    group: str
    probability: float
    confidence_low: float
    confidence_high: float


class PredictionResponse(BaseModel):
    prediction_id: str
    labels: List[PredictionLabel]
    ecg_channels: int
    ecg_length: int
    tabular_features: int
    processing_time_ms: float
    model_used: Optional[str] = None
    demo_mode: bool = False


# ── Models ───────────────────────────────────────────────────────────────────

class ModelCheckpoint(BaseModel):
    checkpoint_id: str
    run_id: str
    path: str
    created_at: str
    epoch: Optional[int] = None
    val_auroc: Optional[float] = None
    is_best: bool = False
    size_mb: float = 0.0


class ModelLoadResponse(BaseModel):
    checkpoint_id: str
    loaded: bool
    message: str


# ── Data ─────────────────────────────────────────────────────────────────────

class ECGUploadResponse(BaseModel):
    file_id: str
    filename: str
    channels: int
    length: int
    fs: float
    duration_seconds: float
    message: str


class ECGWaveformData(BaseModel):
    file_id: str
    channels: int
    length: int
    fs: float
    duration_seconds: float
    lead_names: List[str]
    # each channel is a list of [time, amplitude] pairs — only first N points returned for perf
    data: Dict[str, List[float]]
    time_axis: List[float]


class TabularUploadResponse(BaseModel):
    file_id: str
    filename: str
    num_features: int
    message: str


class DataSample(BaseModel):
    sample_id: str
    description: str
    ecg_channels: int
    ecg_duration_s: float
    num_tabular_features: int
    has_labels: bool


# ── Metrics ──────────────────────────────────────────────────────────────────

class MetricPoint(BaseModel):
    step: int
    value: float


class RunMetrics(BaseModel):
    run_id: str
    train_loss: List[MetricPoint] = []
    val_loss: List[MetricPoint] = []
    val_auroc: List[MetricPoint] = []
    val_aupr: List[MetricPoint] = []
    per_class_auroc: Dict[str, float] = {}
    best_val_auroc: Optional[float] = None
    best_epoch: Optional[int] = None


# ── Config ───────────────────────────────────────────────────────────────────

class ConfigOption(BaseModel):
    key: str
    label: str
    options: List[str]
    current: str


class FullConfigResponse(BaseModel):
    yaml_content: str
    parsed: Dict[str, Any]
    options: List[ConfigOption]


# ── System ───────────────────────────────────────────────────────────────────

class SystemInfo(BaseModel):
    gpu_available: bool
    gpu_count: int
    gpu_names: List[str]
    cpu_count: int
    total_memory_gb: float
    python_version: str
    torch_version: str
    cuda_version: Optional[str]
    backend_root: str
    active_model: Optional[str]
    active_runs: int
