"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  listPatients, predictPatient, getPatientECG,
  type PatientSummary, type PatientPredictionResult, type ECGWaveformData,
} from "@/lib/api";
import { LineChart, Line, ResponsiveContainer, XAxis, YAxis } from "recharts";

const CRITICAL_LABELS = [
  "mortality_inhospital","mortality_30d","detr_cardiac_arrest","detr_icu_transfer",
  "detr_sepsis","detr_respiratory_failure","detr_shock","detr_intubation","detr_cpr",
];

const LABEL_NAMES: Record<string, string> = {
  mortality_inhospital: "In-Hospital Mortality",  mortality_30d: "30-Day Mortality",
  detr_cardiac_arrest: "Cardiac Arrest",          detr_icu_transfer: "ICU Transfer",
  detr_sepsis: "Sepsis",                          detr_respiratory_failure: "Respiratory Failure",
  detr_shock: "Circulatory Shock",                detr_intubation: "Intubation Required",
  detr_cpr: "CPR Required",                       detr_renal_replacement: "Renal Replacement",
  detr_stroke: "Stroke",                          od_aki: "Acute Kidney Injury",
  od_liver_failure: "Liver Failure",              od_coagulopathy: "Coagulopathy",
  od_thrombocytopenia: "Thrombocytopenia",        od_anemia: "Anemia",
  od_electrolyte_imbalance: "Electrolyte Imbalance", od_hyperglycemia: "Hyperglycemia",
  od_hypoglycemia: "Hypoglycemia",                od_hypotension: "Hypotension",
  od_hypertension: "Hypertension",               od_arrhythmia: "Arrhythmia",
  od_heart_failure: "Heart Failure",             od_pneumonia: "Pneumonia",
  od_ards: "ARDS",                               med_vasopressor: "Vasopressor Need",
  med_antibiotic: "Antibiotic Need",             med_sedative: "Sedation Need",
  med_anticoagulant: "Anticoagulant Need",       med_diuretic: "Diuretic Need",
  med_insulin: "Insulin Need",                   med_steroid: "Steroid Need",
  med_antiarrhythmic: "Antiarrhythmic Need",
};

function riskLevel(p: number) {
  if (p >= 0.60) return { label: "HIGH",     color: "text-red-400",    bg: "bg-red-500/15",    border: "border-red-500/50",    bar: "bg-red-500" };
  if (p >= 0.35) return { label: "ELEVATED", color: "text-orange-400", bg: "bg-orange-500/15", border: "border-orange-500/50", bar: "bg-orange-500" };
  if (p >= 0.15) return { label: "LOW",      color: "text-yellow-400", bg: "bg-yellow-500/10", border: "border-yellow-500/40", bar: "bg-yellow-500" };
  return               { label: "MINIMAL",  color: "text-green-400",  bg: "bg-green-500/10",  border: "border-green-500/40",  bar: "bg-green-500" };
}

function VitalBadge({ label, value, unit, warn }: { label: string; value: number | string; unit: string; warn?: boolean }) {
  return (
    <div className={`rounded px-2 py-1.5 text-center ${warn ? "bg-orange-500/15 border border-orange-500/30" : "bg-muted/40"}`}>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className={`text-sm font-bold ${warn ? "text-orange-400" : ""}`}>
        {typeof value === "number" ? value : value}<span className="text-xs font-normal ml-0.5">{unit}</span>
      </p>
    </div>
  );
}

function PatientCard({ patient, onSelect, isSelected }: {
  patient: PatientSummary; onSelect: () => void; isSelected: boolean;
}) {
  const v = patient.vitals;
  const critical = v.sbp < 90 || v.hr > 120 || v.hr < 50 || v.spo2 < 92 || v.map < 65;
  return (
    <div
      onClick={onSelect}
      className={`rounded-xl border p-4 cursor-pointer transition-all hover:shadow-md ${
        isSelected ? "border-primary bg-primary/5 shadow-md" : "border-border hover:border-primary/40"
      } ${critical ? "border-l-4 border-l-red-500" : ""}`}
    >
      <div className="flex items-start gap-3">
        <div className="shrink-0 h-10 w-10 rounded-full bg-gradient-to-br from-blue-500/30 to-purple-500/30 border border-border flex items-center justify-center text-xs font-bold text-muted-foreground">
          {patient.name.split(" ").map(n => n[0]).join("").slice(0, 2)}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <p className="font-semibold text-sm">{patient.name}</p>
            <span className="text-xs text-muted-foreground">{patient.age}y {patient.gender[0]}</span>
            <Badge variant="outline" className="text-xs font-mono">{patient.room}</Badge>
            {critical && <Badge className="text-xs bg-red-600 text-white">Critical</Badge>}
          </div>
          <p className="text-xs text-orange-400 font-medium mt-0.5 truncate">{patient.diagnosis}</p>
          <p className="text-xs text-muted-foreground mt-1 line-clamp-2">{patient.history}</p>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-1.5 mt-3">
        <VitalBadge label="HR" value={v.hr} unit="bpm" warn={v.hr > 120 || v.hr < 50} />
        <VitalBadge label="BP" value={`${v.sbp}/${v.dbp}`} unit="" warn={v.sbp < 90} />
        <VitalBadge label="SpO₂" value={v.spo2} unit="%" warn={v.spo2 < 92} />
        <VitalBadge label="MAP" value={v.map} unit="mmHg" warn={v.map < 65} />
      </div>
      <p className="text-xs text-muted-foreground mt-2 italic truncate">ECG: {patient.ecg_summary}</p>
      <Button size="sm" variant={isSelected ? "default" : "outline"} className="w-full mt-3 text-xs">
        {isSelected ? "Viewing Results" : "Check Detailed Prediction →"}
      </Button>
    </div>
  );
}

export default function PredictPage() {
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [result, setResult] = useState<PatientPredictionResult | null>(null);
  const [ecg, setEcg] = useState<ECGWaveformData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { listPatients().then(setPatients).catch(() => {}); }, []);

  async function runPrediction(patientId: string) {
    setLoading(true); setError(null); setSelected(patientId); setResult(null); setEcg(null);
    try {
      const [pred, ecgData] = await Promise.all([predictPatient(patientId), getPatientECG(patientId, 800)]);
      setResult(pred); setEcg(ecgData);
    } catch (e: unknown) {
      setError(String(e instanceof Error ? e.message : e));
    } finally { setLoading(false); }
  }

  const ecgChartData = ecg
    ? ecg.time_axis.map((t, i) => ({ t: parseFloat(t.toFixed(2)), I: ecg.data["I"]?.[i] ?? 0, II: ecg.data["II"]?.[i] ?? 0 }))
    : [];

  const labels = result?.prediction.labels ?? [];
  const labelMap = Object.fromEntries(labels.map(l => [l.label, l]));
  const perf = result?.model_performance;
  const criticals = CRITICAL_LABELS.map(lbl => labelMap[lbl]).filter(Boolean).sort((a, b) => b.probability - a.probability);
  const overallRisk = criticals.length > 0 ? Math.max(...criticals.map(l => l.probability)) : null;
  const topRisk = overallRisk !== null ? riskLevel(overallRisk) : null;
  const GROUPS = ["Deterioration", "Mortality", "Organ Dysfunction", "Medication"];

  return (
    <div className="space-y-6 max-w-7xl">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h2 className="text-2xl font-bold">ICU Clinical Deterioration Predictor</h2>
          <p className="text-muted-foreground text-sm mt-1">
            Multimodal AI · 12-lead ECG + 1,602 clinical features · Predicts 33 acute outcome labels
          </p>
        </div>
        {perf?.macro_auroc != null && (
          <div className="flex items-center gap-3 px-4 py-2 rounded-lg border border-border bg-muted/20">
            <div>
              <p className="text-xs text-muted-foreground">Model · {perf.run_id}</p>
              <p className="text-xs text-muted-foreground">Test-set AUROC (macro)</p>
            </div>
            <p className="text-2xl font-black text-primary">{(perf.macro_auroc * 100).toFixed(1)}%</p>
            {result && !result.prediction.demo_mode && (
              <Badge className="bg-green-600 text-white text-xs">Live Model</Badge>
            )}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        {/* Patient roster */}
        <div className="xl:col-span-2 space-y-3">
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">ICU Patient Roster</p>
          {patients.length === 0 && <p className="text-xs text-muted-foreground">Loading…</p>}
          {patients.map(p => (
            <PatientCard key={p.patient_id} patient={p} isSelected={selected === p.patient_id} onSelect={() => runPrediction(p.patient_id)} />
          ))}
        </div>

        {/* Results panel */}
        <div className="xl:col-span-3 space-y-4">
          {!result && !loading && (
            <div className="flex flex-col items-center justify-center h-[500px] border border-dashed border-border rounded-xl gap-3">
              <p className="text-4xl">🏥</p>
              <p className="text-muted-foreground text-sm font-medium">Select a patient to view AI prediction</p>
              <p className="text-xs text-muted-foreground max-w-xs text-center">
                Click "Check Detailed Prediction" on any patient card to run the trained model and see 33 clinical outcome probabilities.
              </p>
            </div>
          )}

          {loading && (
            <div className="flex flex-col items-center justify-center h-[500px] border border-dashed border-border rounded-xl gap-3">
              <p className="text-3xl animate-pulse">⚡</p>
              <p className="text-sm font-medium animate-pulse">Running model inference…</p>
              <p className="text-xs text-muted-foreground">S4 sequence encoder + RealMLP tabular encoder + fusion head</p>
            </div>
          )}

          {error && <div className="rounded-lg bg-destructive/10 border border-destructive/30 px-4 py-3 text-sm text-destructive">{error}</div>}

          {result && !loading && (() => {
            const pt = result.patient;
            const pred = result.prediction;
            return (
              <>
                {/* Patient banner */}
                <div className="flex items-center gap-4 p-4 rounded-xl border border-border bg-muted/10">
                  <div className="h-14 w-14 rounded-full bg-gradient-to-br from-blue-500/30 to-purple-500/30 border border-border flex items-center justify-center text-lg font-bold text-muted-foreground shrink-0">
                    {pt.name.split(" ").map((n: string) => n[0]).join("").slice(0, 2)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex flex-wrap gap-2 items-center">
                      <h3 className="font-bold">{pt.name}</h3>
                      <span className="text-sm text-muted-foreground">{pt.age}y {pt.gender}</span>
                      <Badge variant="outline" className="font-mono text-xs">{pt.room}</Badge>
                      {!pred.demo_mode
                        ? <Badge className="bg-green-600 text-white text-xs">Live Inference</Badge>
                        : <Badge variant="secondary" className="text-xs">Demo Mode</Badge>}
                    </div>
                    <p className="text-sm font-semibold text-orange-400 mt-0.5">{pt.diagnosis}</p>
                    <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">{pt.risk_narrative}</p>
                  </div>
                  <div className="text-right shrink-0 text-xs text-muted-foreground space-y-0.5">
                    <p>{pred.processing_time_ms.toFixed(0)}ms</p>
                    <p className="font-mono">{pred.prediction_id}</p>
                  </div>
                </div>

                {/* Risk + ECG row */}
                <div className="grid grid-cols-3 gap-3">
                  <Card className={`${topRisk?.border ?? ""} ${topRisk?.bg ?? ""}`}>
                    <CardContent className="pt-4 pb-3 text-center space-y-1">
                      <p className="text-xs text-muted-foreground uppercase tracking-wide">Overall Risk</p>
                      <p className={`text-5xl font-black ${topRisk?.color ?? ""}`}>
                        {overallRisk !== null ? Math.round(overallRisk * 100) : "—"}<span className="text-base">%</span>
                      </p>
                      <p className={`text-sm font-bold ${topRisk?.color ?? ""}`}>{topRisk?.label}</p>
                      <p className="text-xs text-muted-foreground">{criticals.length} critical indicators</p>
                    </CardContent>
                  </Card>

                  <Card className="col-span-2">
                    <CardHeader className="pb-0 pt-3 px-4">
                      <CardTitle className="text-xs text-muted-foreground">ECG Preview · Leads I (red) & II (blue)</CardTitle>
                    </CardHeader>
                    <CardContent className="px-2 pb-2">
                      <ResponsiveContainer width="100%" height={85}>
                        <LineChart data={ecgChartData}>
                          <XAxis dataKey="t" hide /><YAxis hide />
                          <Line type="monotone" dataKey="I" stroke="#ef4444" dot={false} strokeWidth={1.5} />
                          <Line type="monotone" dataKey="II" stroke="#3b82f6" dot={false} strokeWidth={1.5} />
                        </LineChart>
                      </ResponsiveContainer>
                      <p className="text-xs text-muted-foreground px-2 italic truncate">{pt.ecg_summary}</p>
                    </CardContent>
                  </Card>
                </div>

                {/* Vitals + Labs */}
                <div className="grid grid-cols-2 gap-3">
                  <Card>
                    <CardHeader className="pb-1 pt-3 px-4"><CardTitle className="text-xs">Current Vitals</CardTitle></CardHeader>
                    <CardContent className="px-3 pb-3 grid grid-cols-4 gap-1.5">
                      <VitalBadge label="HR" value={pt.vitals.hr} unit="bpm" warn={pt.vitals.hr > 120 || pt.vitals.hr < 50} />
                      <VitalBadge label="SBP" value={pt.vitals.sbp} unit="mmHg" warn={pt.vitals.sbp < 90} />
                      <VitalBadge label="SpO₂" value={pt.vitals.spo2} unit="%" warn={pt.vitals.spo2 < 92} />
                      <VitalBadge label="MAP" value={pt.vitals.map} unit="mmHg" warn={pt.vitals.map < 65} />
                      <VitalBadge label="RR" value={pt.vitals.rr} unit="/min" warn={pt.vitals.rr > 25} />
                      <VitalBadge label="Temp" value={pt.vitals.temp} unit="°C" warn={pt.vitals.temp > 38.5} />
                      <VitalBadge label="GCS" value={pt.vitals.gcs} unit="/15" warn={pt.vitals.gcs < 12} />
                      <VitalBadge label="DBP" value={pt.vitals.dbp} unit="mmHg" />
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader className="pb-1 pt-3 px-4"><CardTitle className="text-xs">Key Lab Values</CardTitle></CardHeader>
                    <CardContent className="px-3 pb-3 space-y-1.5 text-xs">
                      {Object.entries(pt.labs).slice(0, 6).map(([k, v]) => (
                        <div key={k} className="flex justify-between">
                          <span className="text-muted-foreground capitalize">{k.replace(/_/g, " ")}</span>
                          <span className="font-mono font-semibold">{typeof v === "number" ? v.toFixed(1) : String(v)}</span>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                </div>

                {/* Critical outcomes grid */}
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Critical Outcome Predictions</CardTitle>
                    <CardDescription className="text-xs">
                      Acute deterioration · mortality · cardiac events · ICU transfer
                      {perf?.macro_auroc != null && ` · Model AUROC: ${(perf.macro_auroc * 100).toFixed(1)}%`}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
                      {CRITICAL_LABELS.map(lbl => {
                        const l = labelMap[lbl]; if (!l) return null;
                        const risk = riskLevel(l.probability);
                        const auroc = perf?.per_label_auroc?.[lbl];
                        return (
                          <div key={lbl} className={`rounded-lg border p-3 ${risk.border} ${risk.bg}`}>
                            <p className="text-xs font-semibold">{LABEL_NAMES[lbl] ?? lbl}</p>
                            <div className="flex items-end justify-between mt-1.5">
                              <div>
                                <p className={`text-2xl font-black ${risk.color}`}>{Math.round(l.probability * 100)}%</p>
                                <p className={`text-xs font-bold ${risk.color}`}>{risk.label}</p>
                              </div>
                              {auroc != null && (
                                <div className="text-right">
                                  <p className="text-xs text-muted-foreground">AUROC</p>
                                  <p className="text-xs font-bold text-primary">{(auroc * 100).toFixed(0)}%</p>
                                </div>
                              )}
                            </div>
                            <div className="mt-2 h-1 rounded bg-muted overflow-hidden">
                              <div className={`h-full rounded ${risk.bar}`} style={{ width: `${l.probability * 100}%` }} />
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">
                              CI [{Math.round(l.confidence_low * 100)}, {Math.round(l.confidence_high * 100)}]%
                            </p>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>

                {/* Full breakdown */}
                {GROUPS.map(group => {
                  const gLabels = labels.filter(l => l.group === group).sort((a, b) => b.probability - a.probability);
                  if (!gLabels.length) return null;
                  return (
                    <Card key={group}>
                      <CardHeader className="pb-2"><CardTitle className="text-sm">{group}</CardTitle></CardHeader>
                      <CardContent className="space-y-2">
                        {gLabels.map(l => {
                          const risk = riskLevel(l.probability);
                          const pct = Math.round(l.probability * 100);
                          const auroc = perf?.per_label_auroc?.[l.label];
                          return (
                            <div key={l.label} className="flex items-center gap-2 text-xs">
                              <div className="w-40 shrink-0 font-medium truncate">{LABEL_NAMES[l.label] ?? l.label.replace(/_/g, " ")}</div>
                              <div className="flex-1 h-2 rounded bg-muted overflow-hidden">
                                <div className={`h-full rounded ${risk.bar}`} style={{ width: `${pct}%` }} />
                              </div>
                              <span className={`w-9 text-right font-bold shrink-0 ${risk.color}`}>{pct}%</span>
                              {auroc != null && (
                                <span className="w-14 text-right text-muted-foreground shrink-0">AUC {(auroc * 100).toFixed(0)}%</span>
                              )}
                            </div>
                          );
                        })}
                      </CardContent>
                    </Card>
                  );
                })}
              </>
            );
          })()}
        </div>
      </div>
    </div>
  );
}
