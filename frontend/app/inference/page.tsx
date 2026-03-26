"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
import { predictDemo, predictDemoSample, predictFiles, listSamples, type PredictionResponse, type PredictionLabel, type DataSample } from "@/lib/api";

const GROUP_COLORS: Record<string, string> = {
  Mortality: "text-red-400 bg-red-500/10 border-red-500/30",
  Medication: "text-blue-400 bg-blue-500/10 border-blue-500/30",
  Deterioration: "text-orange-400 bg-orange-500/10 border-orange-500/30",
  "Organ Dysfunction": "text-purple-400 bg-purple-500/10 border-purple-500/30",
};

export default function InferencePage() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ecgFile, setEcgFile] = useState<File | null>(null);
  const [tabFile, setTabFile] = useState<File | null>(null);
  const [activeGroup, setActiveGroup] = useState<string | null>(null);
  const [demoSeed, setDemoSeed] = useState(42);
  const [samples, setSamples] = useState<DataSample[]>([]);
  const [activePatient, setActivePatient] = useState<number | null>(null);

  useEffect(() => {
    listSamples().then(setSamples).catch(() => {});
  }, []);

  async function runPatient(sampleIdx: number) {
    setLoading(true);
    setError(null);
    setActivePatient(sampleIdx);
    try {
      const r = await predictDemoSample(sampleIdx);
      setResult(r);
    } catch (e: unknown) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setLoading(false);
    }
  }

  async function runDemo() {
    setLoading(true);
    setError(null);
    setActivePatient(null);
    try {
      const r = await predictDemo(demoSeed);
      setResult(r);
    } catch (e: unknown) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setLoading(false);
    }
  }

  async function runUpload() {
    if (!ecgFile) { setError("Please upload an ECG file"); return; }
    setLoading(true);
    setError(null);
    try {
      const r = await predictFiles(ecgFile, tabFile ?? undefined);
      setResult(r);
    } catch (e: unknown) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setLoading(false);
    }
  }

  const groups = result
    ? Object.entries(
        result.labels.reduce((acc, l) => {
          if (!acc[l.group]) acc[l.group] = [];
          acc[l.group].push(l);
          return acc;
        }, {} as Record<string, PredictionLabel[]>)
      )
    : [];

  const filteredGroups = activeGroup
    ? groups.filter(([g]) => g === activeGroup)
    : groups;

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h2 className="text-2xl font-bold">Inference</h2>
        <p className="text-muted-foreground text-sm mt-1">
          Get probability predictions for 33 clinical outcome labels
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input panel */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Input Data</CardTitle>
            <CardDescription className="text-xs">Upload files or run a demo</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            {/* Demo Patients */}
            {samples.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs font-medium text-foreground">Demo Patients</p>
                {samples.map((s, i) => (
                  <button
                    key={s.sample_id}
                    onClick={() => runPatient(i)}
                    disabled={loading}
                    className={`w-full text-left text-xs px-2 py-2 rounded border transition-colors ${
                      activePatient === i
                        ? "border-primary bg-primary/10 text-foreground"
                        : "border-border hover:bg-accent text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    <div className="font-medium truncate">{s.sample_id.replace("demo_", "Patient #")}</div>
                    <div className="text-xs text-muted-foreground truncate">{s.description.split(" — ")[1] ?? ""}</div>
                  </button>
                ))}
              </div>
            )}

            <Separator />

            {/* Demo */}
            <div className="space-y-2">
              <p className="text-xs font-medium text-foreground">Synthetic Demo</p>
              <p className="text-xs text-muted-foreground">
                Generates synthetic ECG + tabular data with a random seed.
              </p>
              <div className="flex items-center gap-2">
                <label className="text-xs text-muted-foreground">Seed</label>
                <input
                  type="number"
                  value={demoSeed}
                  onChange={(e) => setDemoSeed(Number(e.target.value))}
                  className="w-20 rounded border border-border bg-background px-2 py-1 text-xs"
                />
              </div>
              <Button onClick={runDemo} disabled={loading} size="sm" className="w-full">
                {loading ? "Running…" : "Run Demo Prediction"}
              </Button>
            </div>

            <Separator />

            {/* Upload */}
            <div className="space-y-3">
              <p className="text-xs font-medium text-foreground">Upload Files</p>
              <FileInput
                label="ECG File (.npy or .csv)*"
                accept=".npy,.csv"
                onChange={setEcgFile}
                value={ecgFile}
              />
              <FileInput
                label="Tabular Features (.npy, .json, .csv)"
                accept=".npy,.json,.csv"
                onChange={setTabFile}
                value={tabFile}
              />
              <Button onClick={runUpload} disabled={loading || !ecgFile} size="sm" variant="outline" className="w-full">
                {loading ? "Running…" : "Predict from Files"}
              </Button>
            </div>

            {error && <p className="text-xs text-destructive">{error}</p>}
          </CardContent>
        </Card>

        {/* Results */}
        <div className="lg:col-span-2 space-y-4">
          {result ? (
            <>
              <div className="flex items-center gap-2">
                {!result.demo_mode ? (
                  <Badge className="bg-green-600 text-white text-xs">Real Model Inference</Badge>
                ) : (
                  <Badge variant="secondary" className="text-xs">Demo Mode (no trained model)</Badge>
                )}
                {activePatient !== null && (
                  <Badge variant="outline" className="text-xs">Patient #{activePatient}</Badge>
                )}
              </div>
              {/* Meta */}
              <Card>
                <CardContent className="pt-4 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <MetaItem label="Prediction ID" value={result.prediction_id} mono />
                  <MetaItem label="ECG" value={`${result.ecg_channels}ch × ${result.ecg_length}pt`} />
                  <MetaItem label="Tabular" value={result.tabular_features > 0 ? `${result.tabular_features} features` : "none"} />
                  <MetaItem label="Time" value={`${result.processing_time_ms.toFixed(1)}ms`} />
                  <MetaItem label="Mode" value={result.demo_mode ? "Demo (no model)" : "Real model"} />
                  <MetaItem label="Model" value={result.model_used ?? "demo fallback"} />
                </CardContent>
              </Card>

              {/* Group filter */}
              <div className="flex flex-wrap gap-2">
                <Button
                  size="sm" variant={activeGroup === null ? "default" : "outline"}
                  onClick={() => setActiveGroup(null)}
                  className="text-xs h-7"
                >All</Button>
                {Object.keys(GROUP_COLORS).map((g) => (
                  <Button
                    key={g} size="sm"
                    variant={activeGroup === g ? "default" : "outline"}
                    onClick={() => setActiveGroup(g === activeGroup ? null : g)}
                    className="text-xs h-7"
                  >{g}</Button>
                ))}
              </div>

              {/* Labels */}
              {filteredGroups.map(([group, labels]) => (
                <Card key={group}>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <span className={`text-xs px-2 py-0.5 rounded border ${GROUP_COLORS[group] ?? ""}`}>{group}</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2.5">
                    {labels.map((l) => (
                      <LabelRow key={l.label} label={l} />
                    ))}
                  </CardContent>
                </Card>
              ))}
            </>
          ) : (
            <div className="flex items-center justify-center h-64 border border-dashed border-border rounded-lg text-muted-foreground text-sm">
              Run a demo or upload files to see predictions
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function LabelRow({ label: l }: { label: PredictionLabel }) {
  const pct = Math.round(l.probability * 100);
  const color =
    pct >= 50 ? "bg-red-500" : pct >= 25 ? "bg-orange-500" : pct >= 10 ? "bg-yellow-500" : "bg-green-500";
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-foreground capitalize">{l.label.replace(/_/g, " ")}</span>
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground text-xs">
            [{l.confidence_low.toFixed(2)}, {l.confidence_high.toFixed(2)}]
          </span>
          <span className="font-medium w-10 text-right">{pct}%</span>
        </div>
      </div>
      <div className="h-1.5 rounded bg-muted overflow-hidden">
        <div className={`h-full rounded ${color} transition-all`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function MetaItem({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <p className="text-muted-foreground mb-0.5">{label}</p>
      <p className={`font-medium truncate ${mono ? "font-mono" : ""}`}>{value}</p>
    </div>
  );
}

function FileInput({
  label, accept, onChange, value,
}: {
  label: string; accept: string; onChange: (f: File | null) => void; value: File | null;
}) {
  return (
    <div>
      <label className="text-xs text-muted-foreground">{label}</label>
      <input
        type="file" accept={accept}
        onChange={(e) => onChange(e.target.files?.[0] ?? null)}
        className="mt-1 block w-full text-xs text-muted-foreground
          file:mr-2 file:py-1 file:px-2 file:rounded file:border file:border-border
          file:bg-background file:text-xs file:text-foreground cursor-pointer"
      />
      {value && <p className="text-xs text-green-400 mt-0.5 truncate">{value.name}</p>}
    </div>
  );
}
