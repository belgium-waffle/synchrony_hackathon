"use client"

import { AlertTriangle, OctagonAlert, RotateCw } from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import type { UnderwriteError } from "@/types/api"

export function VerifierBlockedAlert({ error }: { error: UnderwriteError }) {
  const detail = error.detail
  const failures = detail?.failures ?? []
  return (
    <Alert className="h-full border-rose-500/50 bg-rose-500/[0.06] text-slate-200">
      <OctagonAlert className="text-rose-500" />
      <AlertTitle className="font-mono text-sm font-bold tracking-wide text-rose-400">
        DECISION BLOCKED BY VERIFIER
      </AlertTitle>
      <AlertDescription className="text-slate-300">
        <p className="mb-3">{error.message}</p>
        <ul className="flex flex-col gap-2">
          {failures.map((f, i) => (
            <li
              key={i}
              className="flex items-start gap-2 rounded-md border border-rose-500/30 bg-rose-500/[0.04] p-2 font-mono text-xs text-rose-200"
            >
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-rose-500" />
              <span>{f}</span>
            </li>
          ))}
        </ul>
        <p className="mt-3 text-xs text-slate-400">
          No decision was released. The generated narrative failed deterministic
          validation against the model output.
        </p>
        {detail ? (
          <p className="mt-2 font-mono text-[11px] text-slate-500">
            {detail.verifier_version} · risk_score {detail.risk_score}
          </p>
        ) : null}
      </AlertDescription>
    </Alert>
  )
}

export function GenericErrorAlert({
  error,
  onRetry,
  retryDisabled,
}: {
  error: UnderwriteError
  onRetry: () => void
  retryDisabled: boolean
}) {
  const heading =
    error.kind === "bedrock" ? "LLM SERVICE UNAVAILABLE" : "REQUEST FAILED"
  return (
    <Alert className="border-amber-500/50 bg-amber-500/[0.06] text-slate-200">
      <AlertTriangle className="text-amber-500" />
      <AlertTitle className="font-mono text-sm font-bold tracking-wide text-amber-400">
        {heading}
      </AlertTitle>
      <AlertDescription className="text-slate-300">
        <p className="mb-3">{error.message}</p>
        <Button
          type="button"
          size="sm"
          onClick={onRetry}
          disabled={retryDisabled}
          className="bg-amber-500 font-semibold text-slate-950 hover:bg-amber-400"
        >
          <RotateCw data-icon="inline-start" />
          Retry
        </Button>
      </AlertDescription>
    </Alert>
  )
}
