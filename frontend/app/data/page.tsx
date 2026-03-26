"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  listECGFiles, listTabularFiles, listSamples, uploadECG, uploadTabular, getTabularStats,
  type ECGMeta, type TabularMeta, type DataSample,
} from "@/lib/api";

export default function DataPage() {
  const [ecgFiles, setEcgFiles] = useState<ECGMeta[]>([]);
  const [tabFiles, setTabFiles] = useState<TabularMeta[]>([]);
  const [samples, setSamples] = useState<DataSample[]>([]);
  const [ecgUpload, setEcgUpload] = useState<File | null>(null);
  const [tabUpload, setTabUpload] = useState<File | null>(null);
  const [message, setMessage] = useState<{ text: string; ok: boolean } | null>(null);
  const [tabStats, setTabStats] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = async () => {
    const [ecg, tab, samp] = await Promise.all([
      listECGFiles().catch(() => [] as ECGMeta[]),
      listTabularFiles().catch(() => [] as TabularMeta[]),
      listSamples().catch(() => [] as DataSample[]),
    ]);
    setEcgFiles(ecg);
    setTabFiles(tab);
    setSamples(samp);
  };

  useEffect(() => { refresh(); }, []);

  async function handleECGUpload() {
    if (!ecgUpload) return;
    setLoading(true);
    setMessage(null);
    try {
      const res = await uploadECG(ecgUpload);
      setMessage({ text: res.message, ok: true });
      await refresh();
    } catch (e: unknown) {
      setMessage({ text: String(e instanceof Error ? e.message : e), ok: false });
    } finally {
      setLoading(false);
    }
  }

  async function handleTabUpload() {
    if (!tabUpload) return;
    setLoading(true);
    setMessage(null);
    try {
      const res = await uploadTabular(tabUpload);
      setMessage({ text: res.message, ok: true });
      await refresh();
    } catch (e: unknown) {
      setMessage({ text: String(e instanceof Error ? e.message : e), ok: false });
    } finally {
      setLoading(false);
    }
  }

  async function viewTabStats(fileId: string) {
    const stats = await getTabularStats(fileId).catch(() => null);
    if (stats) setTabStats(stats as unknown as Record<string, unknown>);
  }

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h2 className="text-2xl font-bold">Data Manager</h2>
        <p className="text-muted-foreground text-sm mt-1">Upload and manage ECG waveforms and tabular clinical features</p>
      </div>

      {message && (
        <div className={`rounded-md px-4 py-3 text-sm ${message.ok ? "bg-green-500/10 text-green-400 border border-green-500/30" : "bg-destructive/10 text-destructive border border-destructive/30"}`}>
          {message.text}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* ECG Upload */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">ECG Waveforms</CardTitle>
            <CardDescription className="text-xs">12-lead ECG files (.npy shape 12×N or .csv)</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <input
              type="file" accept=".npy,.csv"
              onChange={(e) => setEcgUpload(e.target.files?.[0] ?? null)}
              className="block w-full text-xs text-muted-foreground file:mr-2 file:py-1 file:px-2 file:rounded file:border file:border-border file:bg-background file:text-xs cursor-pointer"
            />
            <Button size="sm" onClick={handleECGUpload} disabled={!ecgUpload || loading} className="w-full">
              Upload ECG
            </Button>
            <Separator />
            {ecgFiles.length === 0 ? (
              <p className="text-xs text-muted-foreground">No ECG files uploaded yet</p>
            ) : (
              <div className="space-y-1.5">
                {ecgFiles.map((f) => (
                  <div key={f.file_id} className="flex items-center justify-between text-xs p-2 rounded bg-muted/30">
                    <div>
                      <p className="font-medium truncate max-w-[160px]">{f.filename}</p>
                      <p className="text-muted-foreground">{f.channels}ch · {f.duration_seconds.toFixed(1)}s · {f.fs}Hz</p>
                    </div>
                    <Badge variant="outline" className="font-mono text-xs">{f.file_id}</Badge>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Tabular Upload */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Tabular Features</CardTitle>
            <CardDescription className="text-xs">1602 continuous clinical features (.npy, .json, .csv)</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <input
              type="file" accept=".npy,.json,.csv"
              onChange={(e) => setTabUpload(e.target.files?.[0] ?? null)}
              className="block w-full text-xs text-muted-foreground file:mr-2 file:py-1 file:px-2 file:rounded file:border file:border-border file:bg-background file:text-xs cursor-pointer"
            />
            <Button size="sm" variant="outline" onClick={handleTabUpload} disabled={!tabUpload || loading} className="w-full">
              Upload Tabular Data
            </Button>
            <Separator />
            {tabFiles.length === 0 ? (
              <p className="text-xs text-muted-foreground">No tabular files uploaded yet</p>
            ) : (
              <div className="space-y-1.5">
                {tabFiles.map((f) => (
                  <div key={f.file_id} className="flex items-center justify-between text-xs p-2 rounded bg-muted/30 cursor-pointer hover:bg-muted/50"
                    onClick={() => viewTabStats(f.file_id)}>
                    <div>
                      <p className="font-medium truncate max-w-[160px]">{f.filename}</p>
                      <p className="text-muted-foreground">{f.num_features} features</p>
                    </div>
                    <Badge variant="outline" className="font-mono text-xs">{f.file_id}</Badge>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {tabStats && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Tabular Feature Statistics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 text-xs">
              {["mean", "std", "min", "max", "nan_count", "nonzero_count"].map((k) => (
                <div key={k} className="bg-muted/30 rounded p-2">
                  <p className="text-muted-foreground">{k}</p>
                  <p className="font-mono font-semibold mt-0.5">
                    {typeof (tabStats as Record<string, unknown>)?.["stats"] === "object" && (tabStats as Record<string, unknown>)?.["stats"]
                      ? String(((tabStats as Record<string, Record<string, unknown>>)["stats"])[k] ?? "—")
                      : "—"}
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Sample Datasets</CardTitle>
          <CardDescription className="text-xs">Built-in demo scenarios (used by Inference demo mode)</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          {samples.map((s) => (
            <div key={s.sample_id} className="flex items-start justify-between p-3 rounded border border-border">
              <div>
                <p className="text-sm font-medium">{s.sample_id}</p>
                <p className="text-xs text-muted-foreground mt-0.5">{s.description}</p>
                <div className="flex gap-2 mt-1.5">
                  <Badge variant="outline" className="text-xs">{s.ecg_channels}ch ECG</Badge>
                  <Badge variant="outline" className="text-xs">{s.ecg_duration_s}s</Badge>
                  <Badge variant="outline" className="text-xs">{s.num_tabular_features} features</Badge>
                </div>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
