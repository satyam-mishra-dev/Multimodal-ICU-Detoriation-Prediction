"use client";

import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  BarChart, Bar, Cell,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { getDemoMetrics, listMetricRuns, getRunMetrics, type RunMetrics, type MetricRunSummary } from "@/lib/api";

export default function MetricsPage() {
  const [runs, setRuns] = useState<MetricRunSummary[]>([]);
  const [metrics, setMetrics] = useState<RunMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"curves" | "perclass">("curves");

  useEffect(() => {
    listMetricRuns().then(setRuns).catch(() => {});
    loadDemoMetrics();
  }, []);

  async function loadDemoMetrics() {
    setLoading(true);
    try {
      const m = await getDemoMetrics();
      setMetrics(m as RunMetrics);
    } finally {
      setLoading(false);
    }
  }

  async function loadRunMetrics(runId: string) {
    setLoading(true);
    try {
      const m = await getRunMetrics(runId);
      setMetrics(m);
    } catch {
      // fallback to demo
      await loadDemoMetrics();
    } finally {
      setLoading(false);
    }
  }

  const lossData = metrics
    ? metrics.train_loss.map((p, i) => ({
        epoch: p.step,
        "Train Loss": p.value,
        "Val Loss": metrics.val_loss[i]?.value ?? null,
      }))
    : [];

  const aurocData = metrics
    ? metrics.val_auroc.map((p, i) => ({
        epoch: p.step,
        "Val AUROC": p.value,
        "Val AUPR": metrics.val_aupr[i]?.value ?? null,
      }))
    : [];

  const perClassData = metrics
    ? Object.entries(metrics.per_class_auroc)
        .sort((a, b) => b[1] - a[1])
        .map(([label, auroc]) => ({ label: label.replace(/_/g, " "), auroc }))
    : [];

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h2 className="text-2xl font-bold">Metrics</h2>
        <p className="text-muted-foreground text-sm mt-1">Training loss curves, AUROC, AUPR, and per-class performance</p>
      </div>

      <div className="flex flex-wrap gap-2 items-center">
        <Button size="sm" variant="outline" onClick={loadDemoMetrics} disabled={loading}>
          Load Demo Metrics
        </Button>
        {runs.length > 0 && runs.map((r) => (
          <Button key={r.run_id} size="sm" variant="ghost" onClick={() => loadRunMetrics(r.run_id)}>
            {r.run_id}
          </Button>
        ))}
      </div>

      {metrics && (
        <>
          {/* Summary row */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <StatCard label="Best Val AUROC" value={metrics.best_val_auroc?.toFixed(4) ?? "—"} />
            <StatCard label="Best Epoch" value={metrics.best_epoch != null ? String(metrics.best_epoch) : "—"} />
            <StatCard label="Total Epochs" value={String(metrics.val_auroc.length)} />
            <StatCard label="Per-Class Labels" value={String(Object.keys(metrics.per_class_auroc).length)} />
          </div>

          {/* Tab selector */}
          <div className="flex gap-2">
            {(["curves", "perclass"] as const).map((tab) => (
              <Button key={tab} size="sm" variant={activeTab === tab ? "default" : "outline"}
                onClick={() => setActiveTab(tab)}>
                {tab === "curves" ? "Training Curves" : "Per-Class AUROC"}
              </Button>
            ))}
          </div>

          {activeTab === "curves" && (
            <div className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Loss Curves</CardTitle>
                  <CardDescription className="text-xs">Train and validation loss over epochs</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={lossData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis dataKey="epoch" tick={{ fontSize: 11 }} label={{ value: "Epoch", position: "insideBottom", offset: -2, fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} />
                      <Tooltip contentStyle={{ fontSize: 11, background: "#1a1a1a", border: "1px solid #333" }} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line type="monotone" dataKey="Train Loss" stroke="#3b82f6" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="Val Loss" stroke="#f97316" dot={false} strokeWidth={2} strokeDasharray="4 2" />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base">AUROC & AUPR</CardTitle>
                  <CardDescription className="text-xs">Validation area under curves over epochs</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={aurocData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis dataKey="epoch" tick={{ fontSize: 11 }} label={{ value: "Epoch", position: "insideBottom", offset: -2, fontSize: 11 }} />
                      <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                      <Tooltip contentStyle={{ fontSize: 11, background: "#1a1a1a", border: "1px solid #333" }} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line type="monotone" dataKey="Val AUROC" stroke="#22c55e" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="Val AUPR" stroke="#a855f7" dot={false} strokeWidth={2} strokeDasharray="4 2" />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          )}

          {activeTab === "perclass" && perClassData.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Per-Class AUROC</CardTitle>
                <CardDescription className="text-xs">Best validation AUROC for each clinical outcome label</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={Math.max(300, perClassData.length * 28)}>
                  <BarChart data={perClassData} layout="vertical" margin={{ left: 160 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10 }} />
                    <YAxis type="category" dataKey="label" tick={{ fontSize: 10 }} width={155} />
                    <Tooltip formatter={(v) => typeof v === "number" ? v.toFixed(4) : v} contentStyle={{ fontSize: 11, background: "#1a1a1a", border: "1px solid #333" }} />
                    <Bar dataKey="auroc" radius={[0, 3, 3, 0]}>
                      {perClassData.map((d, i) => (
                        <Cell key={i} fill={d.auroc >= 0.85 ? "#22c55e" : d.auroc >= 0.75 ? "#3b82f6" : d.auroc >= 0.65 ? "#f97316" : "#ef4444"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div className="flex gap-4 mt-3 text-xs text-muted-foreground">
                  <span><span className="inline-block w-3 h-3 rounded bg-green-500 mr-1" />≥0.85</span>
                  <span><span className="inline-block w-3 h-3 rounded bg-blue-500 mr-1" />≥0.75</span>
                  <span><span className="inline-block w-3 h-3 rounded bg-orange-500 mr-1" />≥0.65</span>
                  <span><span className="inline-block w-3 h-3 rounded bg-red-500 mr-1" />&lt;0.65</span>
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardContent className="pt-4 pb-3">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="text-xl font-bold mt-1">{value}</p>
      </CardContent>
    </Card>
  );
}
