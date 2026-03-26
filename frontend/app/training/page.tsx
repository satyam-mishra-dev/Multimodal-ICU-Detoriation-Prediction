"use client";

import { useEffect, useState, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  startTraining, listRuns, getRun, getRunLogs, stopRun, deleteRun,
  type TrainingConfig, type TrainingRunSummary, type TrainingRun,
  DEFAULT_TRAINING_CONFIG,
} from "@/lib/api";

export default function TrainingPage() {
  const [runs, setRuns] = useState<TrainingRunSummary[]>([]);
  const [selectedRun, setSelectedRun] = useState<TrainingRun | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [cfg, setCfg] = useState<TrainingConfig>(DEFAULT_TRAINING_CONFIG);
  const [launching, setLaunching] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const logsRef = useRef<HTMLDivElement>(null);

  const refresh = () =>
    listRuns().then(setRuns).catch(() => {});

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (!selectedRun) return;
    const t = setInterval(() => {
      getRun(selectedRun.run_id).then(setSelectedRun).catch(() => {});
      getRunLogs(selectedRun.run_id, 200).then((r) => setLogs(r.logs)).catch(() => {});
    }, 2000);
    return () => clearInterval(t);
  }, [selectedRun?.run_id]);

  useEffect(() => {
    if (logsRef.current) logsRef.current.scrollTop = logsRef.current.scrollHeight;
  }, [logs]);

  async function handleStart() {
    setLaunching(true);
    setMessage(null);
    try {
      const res = await startTraining(cfg);
      setMessage(`Started run ${res.run_id}`);
      refresh();
    } catch (e: unknown) {
      setMessage(`Error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setLaunching(false);
    }
  }

  async function handleStop(runId: string) {
    await stopRun(runId).catch(() => {});
    refresh();
  }

  async function handleDelete(runId: string) {
    await deleteRun(runId).catch(() => {});
    if (selectedRun?.run_id === runId) setSelectedRun(null);
    refresh();
  }

  function handleSelectRun(r: TrainingRunSummary) {
    getRun(r.run_id).then(setSelectedRun);
    getRunLogs(r.run_id, 200).then((res) => setLogs(res.logs));
  }

  return (
    <div className="space-y-6 max-w-6xl">
      <div>
        <h2 className="text-2xl font-bold">Training</h2>
        <p className="text-muted-foreground text-sm mt-1">Configure and launch model training runs</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Config form */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="text-base">Training Config</CardTitle>
            <CardDescription className="text-xs">Hydra overrides for main_all.py</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <ConfigRow label="Epochs">
              <input type="number" value={cfg.epochs} min={1} max={1000}
                onChange={(e) => setCfg({ ...cfg, epochs: Number(e.target.value) })}
                className="w-full rounded border border-border bg-background px-2 py-1 text-sm" />
            </ConfigRow>
            <ConfigRow label="Batch Size">
              <input type="number" value={cfg.batch_size} min={1} max={512}
                onChange={(e) => setCfg({ ...cfg, batch_size: Number(e.target.value) })}
                className="w-full rounded border border-border bg-background px-2 py-1 text-sm" />
            </ConfigRow>
            <ConfigRow label="Learning Rate">
              <input type="number" value={cfg.learning_rate} step={0.0001} min={0.00001}
                onChange={(e) => setCfg({ ...cfg, learning_rate: Number(e.target.value) })}
                className="w-full rounded border border-border bg-background px-2 py-1 text-sm" />
            </ConfigRow>
            <ConfigRow label="GPUs">
              <input type="number" value={cfg.gpus} min={0} max={8}
                onChange={(e) => setCfg({ ...cfg, gpus: Number(e.target.value) })}
                className="w-full rounded border border-border bg-background px-2 py-1 text-sm" />
            </ConfigRow>
            <ConfigRow label="Precision">
              <select value={cfg.precision} onChange={(e) => setCfg({ ...cfg, precision: Number(e.target.value) })}
                className="w-full rounded border border-border bg-background px-2 py-1 text-sm">
                <option value={32}>32-bit</option>
                <option value={16}>16-bit (AMP)</option>
              </select>
            </ConfigRow>
            <ConfigRow label="Predictor">
              <select value={cfg.predictor} onChange={(e) => setCfg({ ...cfg, predictor: e.target.value })}
                className="w-full rounded border border-border bg-background px-2 py-1 text-sm">
                <option value="s4">S4 (State Space)</option>
                <option value="none">None (Identity)</option>
              </select>
            </ConfigRow>
            <ConfigRow label="Loss">
              <select value={cfg.loss} onChange={(e) => setCfg({ ...cfg, loss: e.target.value })}
                className="w-full rounded border border-border bg-background px-2 py-1 text-sm">
                <option value="bcef">Focal BCE</option>
                <option value="bce">BCE</option>
                <option value="ce">Cross-Entropy</option>
                <option value="cef">Focal CE</option>
              </select>
            </ConfigRow>
            <ConfigRow label="Fusion">
              <select value={cfg.fusion} onChange={(e) => setCfg({ ...cfg, fusion: e.target.value })}
                className="w-full rounded border border-border bg-background px-2 py-1 text-sm">
                <option value="poolconcat">Pool + Concat</option>
                <option value="concat">Concat</option>
                <option value="tensor">Tensor Fusion</option>
                <option value="attention">Attention</option>
              </select>
            </ConfigRow>
            <ConfigRow label="S4 Model Dim">
              <input type="number" value={cfg.model_dim} step={64}
                onChange={(e) => setCfg({ ...cfg, model_dim: Number(e.target.value) })}
                className="w-full rounded border border-border bg-background px-2 py-1 text-sm" />
            </ConfigRow>
            <ConfigRow label="S4 State Dim">
              <input type="number" value={cfg.state_dim}
                onChange={(e) => setCfg({ ...cfg, state_dim: Number(e.target.value) })}
                className="w-full rounded border border-border bg-background px-2 py-1 text-sm" />
            </ConfigRow>
            <ConfigRow label="Run Label">
              <input type="text" value={cfg.metadata}
                onChange={(e) => setCfg({ ...cfg, metadata: e.target.value })}
                className="w-full rounded border border-border bg-background px-2 py-1 text-sm" />
            </ConfigRow>
            <Separator />
            {message && (
              <p className={`text-xs ${message.startsWith("Error") ? "text-destructive" : "text-green-400"}`}>
                {message}
              </p>
            )}
            <div className="flex gap-2">
              <Button onClick={handleStart} disabled={launching} size="sm" className="flex-1">
                {launching ? "Starting…" : "Start Training"}
              </Button>
              <Button variant="outline" size="sm" onClick={() => setCfg(DEFAULT_TRAINING_CONFIG)}>
                Reset
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Runs list + logs */}
        <div className="lg:col-span-2 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Runs</CardTitle>
            </CardHeader>
            <CardContent>
              {runs.length === 0 ? (
                <p className="text-sm text-muted-foreground">No training runs yet. Start one on the left.</p>
              ) : (
                <div className="space-y-2">
                  {runs.map((r) => (
                    <div
                      key={r.run_id}
                      onClick={() => handleSelectRun(r)}
                      className={`flex items-center justify-between rounded-md border px-3 py-2 cursor-pointer text-sm transition-colors ${
                        selectedRun?.run_id === r.run_id ? "border-primary bg-primary/5" : "border-border hover:bg-accent"
                      }`}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <StatusBadge status={r.status} />
                        <span className="font-mono text-xs truncate">{r.run_id}</span>
                        <span className="text-muted-foreground text-xs hidden sm:inline">
                          {new Date(r.created_at).toLocaleTimeString()}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5 shrink-0">
                        {r.best_auroc != null && (
                          <span className="text-xs text-muted-foreground">AUROC {r.best_auroc.toFixed(3)}</span>
                        )}
                        {r.status === "running" && (
                          <Button size="sm" variant="destructive" className="h-6 px-2 text-xs"
                            onClick={(e) => { e.stopPropagation(); handleStop(r.run_id); }}>
                            Stop
                          </Button>
                        )}
                        {["completed", "failed", "stopped"].includes(r.status) && (
                          <Button size="sm" variant="ghost" className="h-6 px-2 text-xs text-muted-foreground"
                            onClick={(e) => { e.stopPropagation(); handleDelete(r.run_id); }}>
                            Delete
                          </Button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {selectedRun && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  Logs — <span className="font-mono text-xs">{selectedRun.run_id}</span>
                  <StatusBadge status={selectedRun.status} />
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div
                  ref={logsRef}
                  className="h-64 overflow-y-auto rounded bg-black/40 p-3 font-mono text-xs text-green-300 space-y-0.5"
                >
                  {logs.length === 0 ? (
                    <p className="text-muted-foreground">Waiting for logs…</p>
                  ) : (
                    logs.map((l, i) => <div key={i}>{l}</div>)
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

function ConfigRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2">
      <label className="w-28 shrink-0 text-xs text-muted-foreground">{label}</label>
      <div className="flex-1">{children}</div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, string> = {
    running: "bg-blue-500/20 text-blue-400",
    completed: "bg-green-500/20 text-green-400",
    failed: "bg-red-500/20 text-red-400",
    stopped: "bg-yellow-500/20 text-yellow-400",
    pending: "bg-gray-500/20 text-gray-400",
  };
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded font-mono ${map[status] ?? "bg-muted text-muted-foreground"}`}>
      {status}
    </span>
  );
}
