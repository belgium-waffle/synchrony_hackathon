"use client"

import { useEffect, useState } from "react"
import { Check, Loader2 } from "lucide-react"

import { cn } from "@/lib/utils"

const STAGES = [
  "Stream A — scoring and TreeSHAP",
  "Stream B — retrieving policy from pgvector",
  "Synthesising audit narrative",
  "Running deterministic verification",
]

const STEP_MS = 900

export function LoadingStages() {
  const [active, setActive] = useState(0)

  useEffect(() => {
    const timer = setInterval(() => {
      setActive((prev) => Math.min(prev + 1, STAGES.length - 1))
    }, STEP_MS)
    return () => clearInterval(timer)
  }, [])

  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm font-semibold text-slate-100">
        Running dual-stream analysis
      </p>
      <ol className="flex flex-col gap-2.5">
        {STAGES.map((stage, i) => {
          const done = i < active
          const isActive = i === active
          return (
            <li key={stage} className="flex items-center gap-2.5 text-sm">
              <span className="flex size-5 shrink-0 items-center justify-center">
                {done ? (
                  <Check className="size-4 text-emerald-500" aria-hidden />
                ) : isActive ? (
                  <Loader2
                    className="size-4 animate-spin text-amber-500"
                    aria-hidden
                  />
                ) : (
                  <span
                    className="size-1.5 rounded-full bg-slate-600"
                    aria-hidden
                  />
                )}
              </span>
              <span
                className={cn(
                  done && "text-slate-400",
                  isActive && "text-slate-100",
                  !done && !isActive && "text-slate-600",
                )}
              >
                {stage}
              </span>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
