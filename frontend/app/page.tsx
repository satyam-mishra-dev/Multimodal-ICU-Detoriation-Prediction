"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Activity, Brain, ChartLine, Cpu, Database, Heart, Settings } from "lucide-react";
import { getSystemInfo, listRuns, listModels, listSamples, type SystemInfo, type TrainingRunSummary, type ModelCheckpoint, type DataSample } from "@/lib/api";

export default function DashboardPage() {
  const [sysInfo, setSysInfo] = useState<SystemInfo | null>(null);
  const [runs, setRuns] = useState<TrainingRunSummary[]>([]);
  const [models, setModels] = useState<ModelCheckpoint[]>([]);
  const [samples, setSamples] = useState<DataSample[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getSystemInfo()
      .then(setSysInfo)
      .catch(() => setError("API unreachable — run ./start_api.sh in the repo root"));
    listRuns().then(setRuns).catch(() => {});
    listModels().then(setModels).catch(() => {});
    listSamples().then(setSamples).catch(() => {});
  }, []);

  const runningCount = runs.filter((r) => r.status === "running").length;
  const bestModels = models.filter((m) => m.is_best);

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h2 className="text-2xl font-bold">Dashboard</h2>
        <p className="text-muted-foreground text-sm mt-1">
          Medical Decision Support in ICU — multimodal deep learning framework
        </p>
      </div>

      {error && (
        <div className="rounded-md bg-destructive/15 border border-destructive/30 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {sysInfo && (<>
          <StatCard
            label="Compute"
            value={sysInfo.gpu_available ? `${sysInfo.gpu_count}× GPU` : "CPU"}
            sub={sysInfo.gpu_names[0] ?? `PyTorch ${sysInfo.torch_version}`}
          />
          <StatCard label="Memory" value={`${sysInfo.total_memory_gb} GB`} sub={`${sysInfo.cpu_count} vCPUs`} />
        </>)}
        <StatCard
          label="Trained Models"
          value={bestModels.length > 0 ? `${bestModels.length} checkpoint${bestModels.length > 1 ? "s" : ""}` : "None"}
          sub={bestModels.length > 0 ? `Latest: ${bestModels[bestModels.length - 1]?.run_id ?? "—"}` : "Run train_demo.sh"}
          highlight={bestModels.length > 0}
        />
        <StatCard
          label="Demo Patients"
          value={samples.length > 0 ? `${samples.length} ICU patients` : "Loading…"}
          sub={samples.length > 0 ? "Real ECG + 1602 features" : ""}
          highlight={samples.length > 0}
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <FeatureCard
          href="/inference"
          icon={<Activity className="h-5 w-5" />}
          title="Inference"
          description="Select a demo ICU patient and get real-time probability predictions for 33 clinical outcome labels from the trained model."
          badge="Live"
        />
        <FeatureCard
          href="/ecg"
          icon={<Heart className="h-5 w-5" />}
          title="ECG Viewer"
          description="Explore 12-lead ECG waveforms from 5 demo ICU patients. Real data, visualized per-lead."
          badge="Real data"
        />
        <FeatureCard
          href="/metrics"
          icon={<ChartLine className="h-5 w-5" />}
          title="Metrics"
          description="View AUROC, AUPR, and loss curves across training runs. Includes per-class breakdowns."
        />
        <FeatureCard
          href="/models"
          icon={<Cpu className="h-5 w-5" />}
          title="Models"
          description="Browse saved checkpoints. The latest checkpoint auto-loads for inference on first request."
          badge={bestModels.length > 0 ? `${bestModels.length} ready` : undefined}
        />
        <FeatureCard
          href="/data"
          icon={<Database className="h-5 w-5" />}
          title="Data"
          description="5 ICU demo patients with 12-lead ECG + 1602 tabular clinical features. Upload your own data."
          badge={samples.length > 0 ? `${samples.length} patients` : undefined}
        />
        <FeatureCard
          href="/training"
          icon={<Brain className="h-5 w-5" />}
          title="Training"
          description="Configure and launch training. Monitor live logs, stop runs, and track progress."
          badge={runningCount > 0 ? `${runningCount} running` : undefined}
        />
      </div>

      {runs.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Recent Training Runs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {runs.slice(0, 5).map((r) => (
                <div key={r.run_id} className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <StatusDot status={r.status} />
                    <span className="font-mono text-xs">{r.run_id}</span>
                  </div>
                  <span className="text-muted-foreground text-xs">
                    {r.best_auroc != null ? `AUROC ${r.best_auroc.toFixed(3)}` : "—"}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Architecture Summary</CardTitle>
          <CardDescription>MDS-ICU multimodal pipeline</CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-1.5 text-sm text-muted-foreground">
          {[
            ["Input", "12-lead ECG (250 samples @ 100Hz) + 1602 tabular features"],
            ["Sequence Model", "S4 (State Space) — d=512, 4 layers, bidirectional"],
            ["Tabular Encoder", "RealMLP with quantile normalization → 64-dim"],
            ["Fusion", "Pool + Concatenate"],
            ["Outputs", "33 clinical labels across 4 groups"],
            ["Loss", "Focal BCE (γ=2), NaN-aware"],
            ["Metrics", "AUROC, AUPR with 1000-iter bootstrap CIs"],
            ["Dataset", "MIMIC-IV + MIMIC-IV-ECG"],
          ].map(([k, v]) => (
            <p key={k}>
              <span className="text-foreground font-medium">{k}: </span>{v}
            </p>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}

function StatCard({ label, value, sub, highlight }: { label: string; value: string; sub: string; highlight?: boolean }) {
  return (
    <Card className={highlight ? "border-green-500/40" : ""}>
      <CardContent className="pt-4 pb-3">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className={`text-base font-semibold mt-0.5 truncate ${highlight ? "text-green-400" : ""}`}>{value}</p>
        <p className="text-xs text-muted-foreground mt-0.5 truncate">{sub}</p>
      </CardContent>
    </Card>
  );
}

function FeatureCard({
  href, icon, title, description, badge,
}: {
  href: string; icon: React.ReactNode; title: string; description: string; badge?: string;
}) {
  return (
    <Link href={href}>
      <Card className="h-full hover:border-primary/50 transition-colors cursor-pointer">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div className="text-primary">{icon}</div>
            {badge && <Badge className="text-xs">{badge}</Badge>}
          </div>
          <CardTitle className="text-sm">{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground">{description}</p>
        </CardContent>
      </Card>
    </Link>
  );
}

function StatusDot({ status }: { status: string }) {
  const colors: Record<string, string> = {
    running: "bg-blue-400",
    completed: "bg-green-400",
    failed: "bg-red-400",
    stopped: "bg-yellow-400",
    pending: "bg-gray-400",
  };
  return <span className={`inline-block h-2 w-2 rounded-full ${colors[status] ?? "bg-gray-400"}`} />;
}
