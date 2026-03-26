"use client";

import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { getDemoSampleECG, getDemoECG, getECGWaveform, uploadECG, listECGFiles, listSamples, type ECGWaveformData, type ECGMeta, type DataSample } from "@/lib/api";

const LEAD_COLORS = [
  "#ef4444","#f97316","#eab308","#22c55e",
  "#06b6d4","#3b82f6","#8b5cf6","#ec4899",
  "#14b8a6","#84cc16","#f59e0b","#6366f1",
];

type DemoVariant = "12lead" | "single" | "bipolar";

export default function ECGViewerPage() {
  const [waveform, setWaveform] = useState<ECGWaveformData | null>(null);
  const [uploads, setUploads] = useState<ECGMeta[]>([]);
  const [samples, setSamples] = useState<DataSample[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [visibleLeads, setVisibleLeads] = useState<Set<string>>(new Set());
  const [demoVariant, setDemoVariant] = useState<DemoVariant>("12lead");
  const [demoSeed, setDemoSeed] = useState(42);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [activeSample, setActiveSample] = useState<string | null>(null);
  const [waveformSource, setWaveformSource] = useState<string>("synthetic");

  const loadSampleECG = async (sampleId: string) => {
    setLoading(true);
    setError(null);
    setActiveSample(sampleId);
    try {
      // Try real sample endpoint first
      const data = await getDemoSampleECG(sampleId);
      setWaveform(data);
      setVisibleLeads(new Set(data.lead_names));
      setWaveformSource((data as Record<string, unknown>).source as string ?? "real_demo");
    } catch {
      // fallback to demo endpoint with numeric seed
      const idx = parseInt(sampleId.replace("demo_", ""), 10);
      try {
        const data = await getDemoECG(demoVariant, 42, idx);
        setWaveform(data);
        setVisibleLeads(new Set(data.lead_names));
        setWaveformSource((data as Record<string, unknown>).source as string ?? "demo");
      } catch (e2: unknown) {
        setError(String(e2 instanceof Error ? e2.message : e2));
      }
    } finally {
      setLoading(false);
    }
  };

  const loadDemo = async (variant: DemoVariant = demoVariant, seed = demoSeed) => {
    setLoading(true);
    setError(null);
    setActiveSample(null);
    try {
      const data = await getDemoECG(variant, seed);
      setWaveform(data);
      setVisibleLeads(new Set(data.lead_names));
      setWaveformSource((data as Record<string, unknown>).source as string ?? "demo");
    } catch (e: unknown) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setLoading(false);
    }
  };

  const loadUploads = () => listECGFiles().then(setUploads).catch(() => {});

  useEffect(() => {
    listSamples().then(setSamples).catch(() => {});
    loadUploads();
    // Auto-load first demo sample
    getDemoSampleECG("demo_0").then((data) => {
      setWaveform(data);
      setVisibleLeads(new Set(data.lead_names));
      setActiveSample("demo_0");
      setWaveformSource((data as Record<string, unknown>).source as string ?? "real_demo");
    }).catch(() => loadDemo());
  }, []);

  async function handleUpload() {
    if (!uploadFile) return;
    setLoading(true);
    setError(null);
    try {
      const res = await uploadECG(uploadFile);
      await loadUploads();
      const data = await getECGWaveform(res.file_id);
      setWaveform(data);
      setVisibleLeads(new Set(data.lead_names));
    } catch (e: unknown) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setLoading(false);
    }
  }

  async function handleSelectUploaded(fileId: string) {
    setLoading(true);
    try {
      const data = await getECGWaveform(fileId);
      setWaveform(data);
      setVisibleLeads(new Set(data.lead_names));
    } finally {
      setLoading(false);
    }
  }

  function toggleLead(lead: string) {
    setVisibleLeads((prev) => {
      const next = new Set(prev);
      if (next.has(lead)) { if (next.size > 1) next.delete(lead); }
      else next.add(lead);
      return next;
    });
  }

  // Build chart data: [{time: 0.01, I: 0.3, II: 0.5, ...}, ...]
  const chartData = waveform
    ? waveform.time_axis.map((t, i) => {
        const point: Record<string, number> = { time: parseFloat(t.toFixed(3)) };
        for (const lead of waveform.lead_names) {
          point[lead] = waveform.data[lead]?.[i] ?? 0;
        }
        return point;
      })
    : [];

  // Split into rows of 3 for stacked mini-charts
  const leadRows: string[][] = [];
  const allLeads = waveform?.lead_names ?? [];
  for (let i = 0; i < allLeads.length; i += 3) {
    leadRows.push(allLeads.slice(i, i + 3));
  }

  return (
    <div className="space-y-6 max-w-6xl">
      <div>
        <h2 className="text-2xl font-bold">ECG Viewer</h2>
        <p className="text-muted-foreground text-sm mt-1">Visualize 12-lead ECG waveforms</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Controls */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Controls</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            {/* Demo Patients */}
            {samples.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs font-medium">Demo Patients</p>
                {samples.map((s) => (
                  <button
                    key={s.sample_id}
                    onClick={() => loadSampleECG(s.sample_id)}
                    className={`w-full text-left text-xs px-2 py-2 rounded border transition-colors ${
                      activeSample === s.sample_id
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
            {/* Demo */}
            <div className="space-y-2 border-t border-border pt-3">
              <p className="text-xs font-medium">Manual Demo</p>
              <select
                value={demoVariant}
                onChange={(e) => setDemoVariant(e.target.value as DemoVariant)}
                className="w-full rounded border border-border bg-background px-2 py-1 text-xs"
              >
                <option value="12lead">12-Lead ECG</option>
                <option value="single">Single Lead</option>
                <option value="bipolar">Bipolar (2 Leads)</option>
              </select>
              <div className="flex items-center gap-2">
                <label className="text-xs text-muted-foreground shrink-0">Seed</label>
                <input
                  type="number" value={demoSeed}
                  onChange={(e) => setDemoSeed(Number(e.target.value))}
                  className="w-full rounded border border-border bg-background px-2 py-1 text-xs"
                />
              </div>
              <Button size="sm" className="w-full text-xs" onClick={() => loadDemo(demoVariant, demoSeed)} disabled={loading}>
                Load Demo
              </Button>
            </div>

            <div className="border-t border-border pt-3 space-y-2">
              <p className="text-xs font-medium">Upload ECG</p>
              <input
                type="file" accept=".npy,.csv"
                onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)}
                className="block w-full text-xs text-muted-foreground file:mr-2 file:py-1 file:px-2 file:rounded file:border file:border-border file:bg-background file:text-xs cursor-pointer"
              />
              <Button size="sm" variant="outline" className="w-full text-xs" onClick={handleUpload} disabled={!uploadFile || loading}>
                Upload & View
              </Button>
            </div>

            {uploads.length > 0 && (
              <div className="border-t border-border pt-3 space-y-1">
                <p className="text-xs font-medium">Uploaded Files</p>
                {uploads.map((u) => (
                  <button key={u.file_id} onClick={() => handleSelectUploaded(u.file_id)}
                    className="w-full text-left text-xs px-2 py-1.5 rounded hover:bg-accent text-muted-foreground hover:text-foreground truncate">
                    {u.filename}
                  </button>
                ))}
              </div>
            )}

            {waveform && (
              <div className="border-t border-border pt-3 space-y-2">
                <p className="text-xs font-medium">Leads</p>
                <div className="flex flex-wrap gap-1">
                  {waveform.lead_names.map((lead, i) => (
                    <button
                      key={lead}
                      onClick={() => toggleLead(lead)}
                      style={{ borderColor: visibleLeads.has(lead) ? LEAD_COLORS[i % LEAD_COLORS.length] : undefined }}
                      className={`text-xs px-1.5 py-0.5 rounded border transition-opacity ${
                        visibleLeads.has(lead) ? "opacity-100" : "opacity-30 border-border"
                      }`}
                    >
                      {lead}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {error && <p className="text-xs text-destructive">{error}</p>}
          </CardContent>
        </Card>

        {/* Waveform display */}
        <div className="lg:col-span-3 space-y-3">
          {waveform && (
            <>
              <div className="flex items-center gap-3 text-xs text-muted-foreground flex-wrap">
                <Badge variant="outline">{waveform.channels} leads</Badge>
                <Badge variant="outline">{waveform.fs} Hz</Badge>
                <Badge variant="outline">{waveform.duration_seconds.toFixed(1)}s</Badge>
                <Badge variant="outline">{waveform.length} samples</Badge>
                {waveformSource === "real_demo" && (
                  <Badge className="bg-green-600 text-white text-xs">Real Demo Data</Badge>
                )}
                {waveformSource === "synthetic" && (
                  <Badge variant="secondary" className="text-xs">Synthetic</Badge>
                )}
              </div>

              {/* Overlay chart (all visible leads) */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">All Leads — Overlay</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer key={waveform.file_id} width="100%" height={200}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis dataKey="time" tick={{ fontSize: 10 }} tickFormatter={(v) => `${v}s`} />
                      <YAxis tick={{ fontSize: 10 }} />
                      <Tooltip formatter={(v) => typeof v === "number" ? v.toFixed(3) : v} labelFormatter={(l) => `t=${l}s`} contentStyle={{ fontSize: 11, background: "#1a1a1a", border: "1px solid #333" }} />
                      {waveform.lead_names.filter((l) => visibleLeads.has(l)).map((lead, i) => (
                        <Line
                          key={lead} type="monotone" dataKey={lead}
                          stroke={LEAD_COLORS[waveform.lead_names.indexOf(lead) % LEAD_COLORS.length]}
                          dot={false} strokeWidth={1.5}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Individual lead charts */}
              {leadRows.map((row, ri) => (
                <div key={ri} className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {row.map((lead) => {
                    const idx = waveform.lead_names.indexOf(lead);
                    const color = LEAD_COLORS[idx % LEAD_COLORS.length];
                    if (!visibleLeads.has(lead)) return null;
                    return (
                      <Card key={lead}>
                        <CardHeader className="pb-1 pt-3 px-3">
                          <CardTitle className="text-xs" style={{ color }}>{lead}</CardTitle>
                        </CardHeader>
                        <CardContent className="px-2 pb-2">
                          <ResponsiveContainer width="100%" height={80}>
                            <LineChart data={chartData}>
                              <XAxis dataKey="time" hide />
                              <YAxis hide />
                              <Tooltip formatter={(v) => typeof v === "number" ? v.toFixed(3) : v} contentStyle={{ fontSize: 10, background: "#1a1a1a", border: "1px solid #333" }} />
                              <Line type="monotone" dataKey={lead} stroke={color} dot={false} strokeWidth={1.5} />
                            </LineChart>
                          </ResponsiveContainer>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              ))}
            </>
          )}
          {!waveform && !loading && (
            <div className="flex items-center justify-center h-48 border border-dashed border-border rounded-lg text-muted-foreground text-sm">
              Load a demo or upload an ECG file
            </div>
          )}
          {loading && (
            <div className="flex items-center justify-center h-48 text-muted-foreground text-sm">
              Loading waveform…
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
