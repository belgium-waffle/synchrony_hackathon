"use client"

import useSWR from "swr"
import { Shield, FlaskConical } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { fetchHealth } from "@/lib/api"
import type { HealthResponse } from "@/types/api"
import { cn } from "@/lib/utils"

function HealthBadge({
  label,
  value,
  tone,
}: {
  label: string
  value: string
  tone: "emerald" | "amber" | "slate"
}) {
  const toneClass = {
    emerald: "border-emerald-500/40 text-emerald-400",
    amber: "border-amber-500/40 text-amber-400",
    slate: "border-slate-700 text-slate-400",
  }[tone]

  return (
    <Badge
      variant="outline"
      className={cn(
        "gap-1.5 bg-slate-900 px-2.5 py-1 font-mono text-[11px] tabular-nums",
        toneClass,
      )}
    >
      <span className="text-slate-500">{label}</span>
      <span
        className={cn(
          "size-1.5 rounded-full",
          tone === "emerald" && "bg-emerald-500",
          tone === "amber" && "bg-amber-500",
          tone === "slate" && "bg-slate-500",
        )}
        aria-hidden
      />
      {value}
    </Badge>
  )
}

export function TopBar({
  onVerifierDemo,
  demoDisabled,
}: {
  onVerifierDemo: () => void
  demoDisabled: boolean
}) {
  const { data: health } = useSWR<HealthResponse>("health", fetchHealth, {
    revalidateOnFocus: false,
    shouldRetryOnError: false,
  })

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-slate-800 bg-slate-950 px-4">
      <div className="flex items-center gap-2.5">
        <span className="flex size-8 items-center justify-center rounded-lg border border-amber-500/40 bg-amber-500/10">
          <Shield className="size-4 text-amber-500" aria-hidden />
        </span>
        <div className="flex flex-col leading-tight">
          <span className="text-sm font-semibold text-slate-100">
            Next-Gen Credit Intelligence
          </span>
          <span className="text-[11px] text-slate-500">
            Synchrony Risk Console
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {health ? (
          <>
            <HealthBadge label="model" value={health.model_version} tone="slate" />
            <HealthBadge
              label="policy"
              value={health.policy_store}
              tone={health.policy_store === "pgvector" ? "emerald" : "amber"}
            />
            <HealthBadge
              label="llm"
              value={health.llm_mode}
              tone={health.llm_mode === "bedrock" ? "emerald" : "slate"}
            />
          </>
        ) : (
          <Badge
            variant="outline"
            className="border-slate-700 bg-slate-900 font-mono text-[11px] text-slate-500"
          >
            connecting to /health…
          </Badge>
        )}
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={onVerifierDemo}
          disabled={demoDisabled}
          className="ml-1 h-7 gap-1.5 text-slate-300 hover:bg-slate-800 hover:text-slate-100"
        >
          <FlaskConical data-icon="inline-start" className="text-amber-500" />
          Demo: Verifier
        </Button>
      </div>
    </header>
  )
}
