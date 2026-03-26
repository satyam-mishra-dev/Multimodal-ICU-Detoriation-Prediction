"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  Brain,
  ChartLine,
  Cpu,
  Database,
  Gauge,
  Heart,
  Settings,
  Stethoscope,
} from "lucide-react";
import { cn } from "@/lib/utils";

const nav = [
  { href: "/predict", label: "Predict", icon: Stethoscope },
  { href: "/", label: "Dashboard", icon: Gauge },
  { href: "/training", label: "Training", icon: Brain },
  { href: "/inference", label: "Inference", icon: Activity },
  { href: "/ecg", label: "ECG Viewer", icon: Heart },
  { href: "/metrics", label: "Metrics", icon: ChartLine },
  { href: "/models", label: "Models", icon: Cpu },
  { href: "/data", label: "Data", icon: Database },
  { href: "/config", label: "Config", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="w-56 shrink-0 border-r border-border bg-card flex flex-col">
      <div className="px-4 py-5 border-b border-border">
        <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">MDS-ICU</p>
        <h1 className="text-sm font-bold mt-0.5">ICU Decision Support</h1>
      </div>
      <nav className="flex-1 py-3 px-2 space-y-0.5">
        {nav.map(({ href, label, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            className={cn(
              "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors",
              pathname === href
                ? "bg-primary text-primary-foreground font-medium"
                : "text-muted-foreground hover:bg-accent hover:text-foreground"
            )}
          >
            <Icon className="h-4 w-4 shrink-0" />
            {label}
          </Link>
        ))}
      </nav>
      <div className="px-4 py-3 border-t border-border text-xs text-muted-foreground">
        API: localhost:8000
      </div>
    </aside>
  );
}
