"use client"

import { Lightbulb, ArrowRight, CheckCircle2 } from "lucide-react"
import type { Recourse } from "@/types/api"
import { cn } from "@/lib/utils"

function formatValue(v: number | null): string {
  if (v === null) return "missing"
  return v.toString()
}

export function RecourseCard({ recourse }: { recourse: Recourse }) {
  // Early return if payload is blocked
  if (!recourse) return null;
  
  const positiveDelta = (recourse.score_delta ?? 0) > 0

  return (
    <section className="flex flex-col gap-3 rounded-xl border border-amber-500/40 bg-amber-500/4 p-4">
      <div className="flex items-center gap-2">
        <Lightbulb className="size-4 text-amber-500" aria-hidden />
        <h3 className="text-sm font-semibold text-slate-100">
          Actionable Recourse
        </h3>
      </div>

      <p
        className={cn(
          "text-sm leading-relaxed",
          recourse.available ? "text-slate-200" : "text-slate-500",
        )}
      >
        {recourse.advice}
      </p>

      {recourse.available ? (
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-slate-800 bg-slate-950/60 p-2.5">
          <span className="text-xs font-medium text-slate-300">
            {recourse.display_name}
          </span>
          <span className="flex items-center gap-1.5 font-mono text-xs text-slate-400 tabular-nums">
            {formatValue(recourse.current_value)}
            <ArrowRight className="size-3 text-slate-600" aria-hidden />
            {formatValue(recourse.target_value)}
          </span>
          {recourse.score_delta !== null ? (
            <span
              className={cn(
                "ml-auto rounded-md border px-2 py-0.5 font-mono text-xs font-semibold tabular-nums",
                positiveDelta
                  ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-400"
                  : "border-slate-700 bg-slate-900 text-slate-400",
              )}
            >
              {positiveDelta ? "+" : ""}
              {recourse.score_delta} pts
            </span>
          ) : null}
        </div>
      ) : null}

      {recourse.crosses_threshold ? (
        <div className="flex items-center gap-2 rounded-lg border border-emerald-500/40 bg-emerald-500/10 p-2.5 text-xs text-emerald-400">
          <CheckCircle2 className="size-4" aria-hidden />
          Would reach {recourse.crosses_threshold}
        </div>
      ) : null}
    </section>
  )
}