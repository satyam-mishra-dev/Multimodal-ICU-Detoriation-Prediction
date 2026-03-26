"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { listModels, getActiveModel, loadModel, unloadModel, type ModelCheckpoint } from "@/lib/api";

export default function ModelsPage() {
  const [checkpoints, setCheckpoints] = useState<ModelCheckpoint[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [message, setMessage] = useState<{ text: string; ok: boolean } | null>(null);

  const refresh = async () => {
    const [ckpts, active] = await Promise.all([
      listModels().catch(() => [] as ModelCheckpoint[]),
      getActiveModel().catch(() => ({ active: false })),
    ]);
    setCheckpoints(ckpts);
    setActiveId("checkpoint_id" in active ? active.checkpoint_id ?? null : null);
  };

  useEffect(() => { refresh(); }, []);

  async function handleLoad(id: string) {
    setLoading(id);
    setMessage(null);
    try {
      const res = await loadModel(id);
      setMessage({ text: res.message, ok: res.loaded });
      await refresh();
    } finally {
      setLoading(null);
    }
  }

  async function handleUnload() {
    setLoading("unload");
    try {
      const res = await unloadModel();
      setMessage({ text: res.message, ok: true });
      setActiveId(null);
    } finally {
      setLoading(null);
    }
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h2 className="text-2xl font-bold">Models</h2>
        <p className="text-muted-foreground text-sm mt-1">Browse saved checkpoints and load a model for inference</p>
      </div>

      {/* Active model status */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Active Model</CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-between">
          {activeId ? (
            <div className="flex items-center gap-3">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-green-400" />
              <div>
                <p className="text-sm font-medium">Model loaded</p>
                <p className="text-xs text-muted-foreground font-mono">{activeId}</p>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-3">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-yellow-400" />
              <div>
                <p className="text-sm font-medium">Demo mode</p>
                <p className="text-xs text-muted-foreground">No checkpoint loaded — inference uses synthetic predictions</p>
              </div>
            </div>
          )}
          {activeId && (
            <Button size="sm" variant="outline" onClick={handleUnload} disabled={loading === "unload"}>
              Unload
            </Button>
          )}
        </CardContent>
      </Card>

      {message && (
        <div className={`rounded-md px-4 py-3 text-sm ${message.ok ? "bg-green-500/10 text-green-400 border border-green-500/30" : "bg-destructive/10 text-destructive border border-destructive/30"}`}>
          {message.text}
        </div>
      )}

      {/* Checkpoint list */}
      {checkpoints.length === 0 ? (
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground text-sm">
            No checkpoints found in the outputs directory.
            <br />
            Train a model first via the <strong>Training</strong> tab.
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-3">
          {checkpoints.map((ckpt) => (
            <Card key={ckpt.checkpoint_id} className={activeId === ckpt.checkpoint_id ? "border-green-500/40" : ""}>
              <CardContent className="pt-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-mono text-xs text-muted-foreground">{ckpt.checkpoint_id}</span>
                    {ckpt.is_best && <Badge variant="outline" className="text-xs border-green-500/50 text-green-400">best</Badge>}
                    {activeId === ckpt.checkpoint_id && <Badge className="text-xs">loaded</Badge>}
                  </div>
                  <p className="text-sm font-medium mt-1 truncate">{ckpt.path.split("/").slice(-3).join("/")}</p>
                  <div className="flex gap-3 mt-1 text-xs text-muted-foreground flex-wrap">
                    <span>{ckpt.size_mb} MB</span>
                    {ckpt.epoch != null && <span>Epoch {ckpt.epoch}</span>}
                    {ckpt.val_auroc != null && <span>AUROC {ckpt.val_auroc.toFixed(4)}</span>}
                    <span>{new Date(ckpt.created_at).toLocaleString()}</span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5 truncate">Run: {ckpt.run_id}</p>
                </div>
                <div className="flex gap-2 shrink-0">
                  {activeId === ckpt.checkpoint_id ? (
                    <Button size="sm" variant="outline" onClick={handleUnload} disabled={loading === "unload"}>
                      Unload
                    </Button>
                  ) : (
                    <Button size="sm" onClick={() => handleLoad(ckpt.checkpoint_id)} disabled={loading === ckpt.checkpoint_id}>
                      {loading === ckpt.checkpoint_id ? "Loading…" : "Load for Inference"}
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">About Checkpoints</CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-muted-foreground space-y-1.5">
          <p>Checkpoints are saved to <code className="font-mono text-xs bg-muted px-1 rounded">ICU-BACKEND/src/outputs/</code> during training by PyTorch Lightning.</p>
          <p>Two files are saved per run: <code className="font-mono text-xs bg-muted px-1 rounded">best_model.ckpt</code> (best validation AUROC) and <code className="font-mono text-xs bg-muted px-1 rounded">last.ckpt</code> (final epoch).</p>
          <p>When no model is loaded, inference runs in <strong>demo mode</strong> which generates realistic synthetic predictions based on the input data.</p>
        </CardContent>
      </Card>
    </div>
  );
}
