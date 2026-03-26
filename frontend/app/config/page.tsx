"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { getCurrentConfig, getArchitectureInfo, getConfigOptions, getConfigPresets, type FullConfigResponse, type ArchitectureInfo } from "@/lib/api";

type Tab = "yaml" | "architecture" | "options" | "presets";

export default function ConfigPage() {
  const [config, setConfig] = useState<FullConfigResponse | null>(null);
  const [arch, setArch] = useState<ArchitectureInfo | null>(null);
  const [tab, setTab] = useState<Tab>("architecture");

  useEffect(() => {
    getCurrentConfig().then(setConfig).catch(() => {});
    getArchitectureInfo().then(setArch).catch(() => {});
  }, []);

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h2 className="text-2xl font-bold">Configuration & Architecture</h2>
        <p className="text-muted-foreground text-sm mt-1">Model architecture, training config, and component options</p>
      </div>

      <div className="flex gap-2 flex-wrap">
        {(["architecture", "yaml", "options", "presets"] as Tab[]).map((t) => (
          <Button key={t} size="sm" variant={tab === t ? "default" : "outline"} onClick={() => setTab(t)}>
            {t === "yaml" ? "YAML Config" : t === "architecture" ? "Architecture" : t === "options" ? "Components" : "Presets"}
          </Button>
        ))}
      </div>

      {tab === "architecture" && arch && (
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Model Pipeline</CardTitle>
              <CardDescription className="text-xs">Step-by-step architecture of the MDS-ICU multimodal model</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="relative">
                {arch.pipeline.map((step, i) => (
                  <div key={step.step} className="flex gap-4 mb-0">
                    <div className="flex flex-col items-center">
                      <div className="w-8 h-8 rounded-full bg-primary/20 border border-primary/40 flex items-center justify-center text-xs font-bold text-primary shrink-0">
                        {step.step}
                      </div>
                      {i < arch.pipeline.length - 1 && (
                        <div className="w-px flex-1 bg-border min-h-4" />
                      )}
                    </div>
                    <div className="pb-5 min-w-0">
                      <p className="text-sm font-semibold">{step.name}</p>
                      <p className="text-xs text-muted-foreground">{step.description}</p>
                      <p className="text-xs text-muted-foreground mt-0.5 font-mono bg-muted/50 rounded px-2 py-1 mt-1.5 whitespace-pre-wrap">
                        {step.details}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Output Labels (33 total)</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {Object.entries(arch.label_groups).map(([group, labels]) => (
                <div key={group}>
                  <p className="text-xs font-semibold text-foreground mb-1.5">{group}</p>
                  <div className="flex flex-wrap gap-1.5">
                    {labels.map((l) => (
                      <Badge key={l} variant="outline" className="text-xs font-mono">
                        {l.replace(/_/g, " ")}
                      </Badge>
                    ))}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      )}

      {tab === "yaml" && config && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">config_supervised_multimodal_mdsicu.yaml</CardTitle>
            <CardDescription className="text-xs">Hydra configuration file — passed to main_all.py</CardDescription>
          </CardHeader>
          <CardContent>
            <pre className="text-xs font-mono bg-black/40 rounded p-4 overflow-x-auto text-green-300 whitespace-pre">
              {config.yaml_content}
            </pre>
          </CardContent>
        </Card>
      )}

      {tab === "options" && (
        <ConfigOptionsTab />
      )}

      {tab === "presets" && (
        <PresetsTab />
      )}
    </div>
  );
}

function ConfigOptionsTab() {
  const [options, setOptions] = useState<Record<string, { label: string; options: { value: string; label: string; description: string }[] }> | null>(null);

  useEffect(() => { getConfigOptions().then(setOptions as never).catch(() => {}); }, []);

  if (!options) return <p className="text-sm text-muted-foreground">Loading…</p>;

  return (
    <div className="space-y-4">
      {Object.entries(options).map(([key, section]) => (
        <Card key={key}>
          <CardHeader>
            <CardTitle className="text-sm">{section.label}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {section.options.map((opt) => (
              <div key={opt.value} className="flex items-start gap-3">
                <Badge variant="outline" className="font-mono text-xs shrink-0">{opt.value}</Badge>
                <div>
                  <p className="text-sm font-medium">{opt.label}</p>
                  <p className="text-xs text-muted-foreground">{opt.description}</p>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function PresetsTab() {
  const [presets, setPresets] = useState<{ id: string; name: string; description: string; config: Record<string, unknown> }[]>([]);

  useEffect(() => {
    getConfigPresets().then((r) => setPresets(r.presets as never)).catch(() => {});
  }, []);

  return (
    <div className="space-y-4">
      {presets.map((p) => (
        <Card key={p.id}>
          <CardHeader>
            <CardTitle className="text-sm">{p.name}</CardTitle>
            <CardDescription className="text-xs">{p.description}</CardDescription>
          </CardHeader>
          <CardContent>
            <pre className="text-xs font-mono bg-black/40 rounded p-3 overflow-x-auto text-blue-300 whitespace-pre">
              {JSON.stringify(p.config, null, 2)}
            </pre>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
