const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── System ─────────────────────────────────────────────────────────────────
export const getSystemInfo = () => req<SystemInfo>("/system/info");
export const getHealth = () => req<{ status: string }>("/system/health");

// ── Training ───────────────────────────────────────────────────────────────
export const startTraining = (cfg: TrainingConfig) =>
  req<TrainingStartResponse>("/training/start", { method: "POST", body: JSON.stringify(cfg) });

export const listRuns = () => req<TrainingRunSummary[]>("/training/runs");

export const getRun = (runId: string) =>
  req<TrainingRun>(`/training/runs/${runId}`);

export const getRunLogs = (runId: string, tail = 100) =>
  req<{ logs: string[]; total_lines: number }>(`/training/runs/${runId}/logs?tail=${tail}`);

export const stopRun = (runId: string) =>
  req<{ message: string }>(`/training/runs/${runId}/stop`, { method: "POST" });

export const deleteRun = (runId: string) =>
  req<{ message: string }>(`/training/runs/${runId}`, { method: "DELETE" });

// ── Inference ──────────────────────────────────────────────────────────────
export const predictDemo = (seed = 42) =>
  req<PredictionResponse>(`/inference/predict/demo?seed=${seed}`, { method: "POST" });

export const predictJson = (ecg: number[][], tabular?: number[]) =>
  req<PredictionResponse>("/inference/predict/json", {
    method: "POST",
    body: JSON.stringify({ ecg, tabular }),
  });

export async function predictFiles(ecgFile: File, tabularFile?: File) {
  const fd = new FormData();
  fd.append("ecg_file", ecgFile);
  if (tabularFile) fd.append("tabular_file", tabularFile);
  const res = await fetch(`${BASE}/inference/predict`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<PredictionResponse>;
}

// ── Data ───────────────────────────────────────────────────────────────────
export const getDemoECG = (config = "12lead", seed = 42, sampleIdx = 0) =>
  req<ECGWaveformData>(`/data/ecg/demo/${config}?seed=${seed}&sample_idx=${sampleIdx}`);

export const getDemoSampleECG = (sampleId: string, maxPoints = 2000) =>
  req<ECGWaveformData>(`/data/ecg/demo_sample/${sampleId}?max_points=${maxPoints}`);

export const listECGFiles = () => req<ECGMeta[]>("/data/ecg");

export const getECGWaveform = (fileId: string, maxPoints = 1000) =>
  req<ECGWaveformData>(`/data/ecg/${fileId}?max_points=${maxPoints}`);

export const listTabularFiles = () => req<TabularMeta[]>("/data/tabular");

export const getTabularStats = (fileId: string) =>
  req<TabularMeta & { stats: Record<string, number> }>(`/data/tabular/${fileId}`);

export const listSamples = () => req<DataSample[]>("/data/samples");

export const predictDemoSample = (sampleIdx = 0) =>
  req<PredictionResponse>(`/inference/predict/demo?sample_idx=${sampleIdx}`, { method: "POST" });

export async function uploadECG(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${BASE}/data/upload/ecg`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<ECGUploadResponse>;
}

export async function uploadTabular(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${BASE}/data/upload/tabular`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<TabularUploadResponse>;
}

export const getModelScores = () => req<ModelScores>("/inference/model_scores");

// ── Patients ────────────────────────────────────────────────────────────────
export const listPatients = () => req<PatientSummary[]>("/patients");
export const predictPatient = (patientId: string) =>
  req<PatientPredictionResult>(`/patients/${patientId}/predict`, { method: "POST" });
export const getPatientECG = (patientId: string, maxPoints = 1000) =>
  req<ECGWaveformData>(`/patients/${patientId}/ecg?max_points=${maxPoints}`);

// ── Models ─────────────────────────────────────────────────────────────────
export const listModels = () => req<ModelCheckpoint[]>("/models");
export const getActiveModel = () => req<{ active: boolean; checkpoint_id?: string }>("/models/active");
export const loadModel = (checkpointId: string) =>
  req<ModelLoadResponse>(`/models/${checkpointId}/load`, { method: "POST" });
export const unloadModel = () =>
  req<{ message: string }>("/models/unload", { method: "POST" });

// ── Metrics ────────────────────────────────────────────────────────────────
export const listMetricRuns = () => req<MetricRunSummary[]>("/metrics/runs");
export const getRunMetrics = (runId: string) =>
  req<RunMetrics>(`/metrics/runs/${runId}`);
export const getDemoMetrics = () => req<RunMetrics>("/metrics/demo");

// ── Config ─────────────────────────────────────────────────────────────────
export const getCurrentConfig = () => req<FullConfigResponse>("/config/current");
export const getConfigOptions = () => req<Record<string, ConfigSection>>("/config/options");
export const getConfigPresets = () => req<{ presets: ConfigPreset[] }>("/config/presets");
export const getArchitectureInfo = () => req<ArchitectureInfo>("/config/architecture");

// ── Types ───────────────────────────────────────────────────────────────────
export interface SystemInfo {
  gpu_available: boolean;
  gpu_count: number;
  gpu_names: string[];
  cpu_count: number;
  total_memory_gb: number;
  python_version: string;
  torch_version: string;
  cuda_version: string | null;
  backend_root: string;
  active_model: string | null;
  active_runs: number;
}

export interface TrainingConfig {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  weight_decay: number;
  gpus: number;
  precision: number;
  predictor: string;
  encoder: string;
  fusion: string;
  loss: string;
  model_dim: number;
  state_dim: number;
  s4_layers: number;
  input_channels: number;
  input_channels_cont: number;
  fs: number;
  input_size: number;
  resume: string;
  eval_only: string;
  metadata: string;
}

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  epochs: 20,
  batch_size: 32,
  learning_rate: 0.001,
  weight_decay: 0.001,
  gpus: 0,
  precision: 32,
  predictor: "s4",
  encoder: "none",
  fusion: "poolconcat",
  loss: "bcef",
  model_dim: 512,
  state_dim: 8,
  s4_layers: 4,
  input_channels: 12,
  input_channels_cont: 1602,
  fs: 100.0,
  input_size: 250,
  resume: "",
  eval_only: "",
  metadata: "run",
};

export interface TrainingStartResponse {
  run_id: string;
  status: string;
  message: string;
}

export interface TrainingRunSummary {
  run_id: string;
  status: string;
  created_at: string;
  updated_at: string;
  epochs_done: number;
  best_auroc: number | null;
}

export interface TrainingRun extends TrainingRunSummary {
  config: TrainingConfig;
  pid: number | null;
  logs: string[];
  metrics: Record<string, unknown>;
}

export interface PredictionLabel {
  label: string;
  group: string;
  probability: number;
  confidence_low: number;
  confidence_high: number;
}

export interface PredictionResponse {
  prediction_id: string;
  labels: PredictionLabel[];
  ecg_channels: number;
  ecg_length: number;
  tabular_features: number;
  processing_time_ms: number;
  model_used: string | null;
  demo_mode: boolean;
}

export interface ECGWaveformData {
  file_id: string;
  channels: number;
  length: number;
  fs: number;
  duration_seconds: number;
  lead_names: string[];
  data: Record<string, number[]>;
  time_axis: number[];
}

export interface ECGMeta {
  file_id: string;
  filename: string;
  channels: number;
  length: number;
  fs: number;
  duration_seconds: number;
  uploaded_at: string;
}

export interface ECGUploadResponse {
  file_id: string;
  filename: string;
  channels: number;
  length: number;
  fs: number;
  duration_seconds: number;
  message: string;
}

export interface TabularMeta {
  file_id: string;
  filename: string;
  num_features: number;
  uploaded_at: string;
}

export interface TabularUploadResponse {
  file_id: string;
  filename: string;
  num_features: number;
  message: string;
}

export interface DataSample {
  sample_id: string;
  description: string;
  ecg_channels: number;
  ecg_duration_s: number;
  num_tabular_features: number;
  has_labels: boolean;
}

export interface ModelCheckpoint {
  checkpoint_id: string;
  run_id: string;
  path: string;
  created_at: string;
  epoch: number | null;
  val_auroc: number | null;
  is_best: boolean;
  size_mb: number;
}

export interface ModelLoadResponse {
  checkpoint_id: string;
  loaded: boolean;
  message: string;
}

export interface MetricPoint {
  step: number;
  value: number;
}

export interface RunMetrics {
  run_id: string;
  train_loss: MetricPoint[];
  val_loss: MetricPoint[];
  val_auroc: MetricPoint[];
  val_aupr: MetricPoint[];
  per_class_auroc: Record<string, number>;
  best_val_auroc: number | null;
  best_epoch: number | null;
}

export interface MetricRunSummary {
  run_id: string;
  source: string;
  status: string;
  created_at: string | null;
  has_tensorboard: boolean;
  metrics_summary: Record<string, unknown>;
}

export interface FullConfigResponse {
  yaml_content: string;
  parsed: Record<string, unknown>;
  options: ConfigOption[];
}

export interface ConfigOption {
  key: string;
  label: string;
  options: string[];
  current: string;
}

export interface ConfigSection {
  label: string;
  options: { value: string; label: string; description: string }[];
}

export interface ConfigPreset {
  id: string;
  name: string;
  description: string;
  config: TrainingConfig;
}

export interface ModelScores {
  available: boolean;
  run_id?: string;
  macro_auroc: number | null;
  per_label: Record<string, number | null>;
}

export interface ArchitectureInfo {
  pipeline: {
    step: number;
    name: string;
    description: string;
    details: string;
  }[];
  label_groups: Record<string, string[]>;
}

export interface PatientVitals {
  hr: number; sbp: number; dbp: number; spo2: number;
  rr: number; temp: number; gcs: number; map: number;
}

export interface PatientSummary {
  patient_id: string;
  name: string;
  age: number;
  gender: string;
  room: string;
  diagnosis: string;
  history: string;
  ecg_summary: string;
  vitals: PatientVitals;
  labs: Record<string, number>;
  risk_narrative: string;
}

export interface PatientPredictionResult {
  patient: PatientSummary;
  prediction: {
    prediction_id: string;
    demo_mode: boolean;
    model_used: string | null;
    processing_time_ms: number;
    ecg_channels: number;
    ecg_length: number;
    tabular_features: number;
    labels: Array<{
      label: string;
      group: string;
      probability: number;
      confidence_low: number;
      confidence_high: number;
    }>;
  };
  model_performance: {
    run_id: string | null;
    macro_auroc: number | null;
    per_label_auroc: Record<string, number | null>;
  };
}
