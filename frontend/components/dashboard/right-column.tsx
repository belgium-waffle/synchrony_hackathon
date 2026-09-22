"use client"

import { ShieldCheck, FileText } from "lucide-react"

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import {
  GenericErrorAlert,
  VerifierBlockedAlert,
} from "@/components/dashboard/error-alerts"
import { formatCurrency, formatMs, formatPd } from "@/lib/format"
import type {
  RetrievedPolicy,
  UnderwriteError,
  UnderwriteResponse,
} from "@/types/api"
import { cn } from "@/lib/utils"

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <Card className="gap-0 border-slate-800 bg-slate-900 py-0">
      <CardHeader className="border-b border-slate-800 px-4 py-3">
        <CardTitle className="text-sm font-semibold tracking-wide text-slate-100 uppercase">
          Compliance Audit Trail
        </CardTitle>
      </CardHeader>
      <CardContent className="px-4 py-4">{children}</CardContent>
    </Card>
  )
}

function KeyValue({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <dt className="text-[11px] text-slate-500">{label}</dt>
      <dd className="font-mono text-xs text-slate-200 tabular-nums">{value}</dd>
    </div>
  )
}

function PolicyBody({ policy }: { policy: RetrievedPolicy }) {
  return (
    <div className="flex flex-col gap-1 rounded-lg border border-slate-800 bg-slate-950/60 p-2.5">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-semibold text-slate-200">
          {policy.title}
        </span>
        <Badge
          variant="outline"
          className="border-amber-500/40 font-mono text-[10px] text-amber-400"
        >
          {policy.policy_id}
        </Badge>
      </div>
      <div className="max-h-32 overflow-y-auto text-xs leading-relaxed text-slate-400">
        {policy.body}
      </div>
    </div>
  )
}

export function RightColumn({
  loading,
  data,
  error,
  hasSubmitted,
  onRetry,
}: {
  loading: boolean
  data: UnderwriteResponse | null
  error: UnderwriteError | null
  hasSubmitted: boolean
  onRetry: () => void
}) {
  if (error?.kind === "verifier_blocked") {
    return <VerifierBlockedAlert error={error} />
  }

  if (error?.kind === "bedrock" || error?.kind === "network") {
    return (
      <Shell>
        <GenericErrorAlert error={error} onRetry={onRetry} retryDisabled={loading} />
      </Shell>
    )
  }

  if (loading) {
    return (
      <Shell>
        <div className="flex flex-col gap-3">
          <Skeleton className="h-12 w-full bg-slate-800" />
          <Skeleton className="h-24 w-full bg-slate-800" />
          <Skeleton className="h-32 w-full bg-slate-800" />
          <Skeleton className="h-20 w-full bg-slate-800" />
        </div>
      </Shell>
    )
  }

  if (!data || !hasSubmitted) {
    return (
      <Shell>
        <div className="flex min-h-64 flex-col items-center justify-center gap-3 text-center">
          <FileText className="size-8 text-slate-700" aria-hidden />
          <p className="max-w-xs text-sm text-slate-500">
            The audit trail, policy citations and verification record appear here
            after an analysis.
          </p>
        </div>
      </Shell>
    )
  }

  const { verification } = data
  const policyById = new Map(
    data.retrieved_policies.map((p) => [p.policy_id, p]),
  )

  return (
    <Shell>
      <div className="flex flex-col gap-4">
        {verification.passed ? (
          <div
            className="flex items-center gap-3 rounded-xl border border-emerald-500/50 bg-emerald-500/10 p-3"
            aria-live="polite"
          >
            <ShieldCheck className="size-6 shrink-0 text-emerald-400" aria-hidden />
            <div className="flex flex-col leading-tight">
              <span className="text-sm font-bold tracking-wide text-emerald-400">
                VERIFIED — {verification.checks_run} deterministic checks passed
              </span>
              <span className="font-mono text-[11px] text-slate-500">
                {verification.verifier_version}
              </span>
            </div>
          </div>
        ) : null}

        <Accordion
          defaultValue={[0, 1, 2, 3, 4]}
          className="flex flex-col gap-1"
        >
          <AccordionItem value={0} className="border-slate-800">
            <AccordionTrigger className="text-slate-200">
              Decision Summary
            </AccordionTrigger>
            <AccordionContent>
              <div className="flex flex-col gap-3">
                <p className="text-xs leading-relaxed text-slate-400">
                  {data.summary}
                </p>
                <dl className="grid grid-cols-2 gap-3">
                  <KeyValue label="Risk Score" value={String(data.risk_score)} />
                  <KeyValue label="PD" value={formatPd(data.probability_of_default)} />
                  <KeyValue label="Status" value={data.approval_status} />
                  <KeyValue
                    label="Recommended Limit"
                    value={formatCurrency(data.recommended_credit_limit)}
                  />
                  <KeyValue label="Model Version" value={data.model_version} />
                </dl>
              </div>
            </AccordionContent>
          </AccordionItem>

          {data.adverse_action_reasons.length > 0 ? (
            <AccordionItem value={1} className="border-slate-800">
              <AccordionTrigger className="text-slate-200">
                Adverse Action Reasons
              </AccordionTrigger>
              <AccordionContent>
                <ul className="flex flex-col gap-2">
                  {data.adverse_action_reasons.map((reason, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs text-slate-300">
                      <span
                        className="mt-1.5 size-1.5 shrink-0 rounded-full bg-rose-500"
                        aria-hidden
                      />
                      {reason}
                    </li>
                  ))}
                </ul>
              </AccordionContent>
            </AccordionItem>
          ) : null}

          <AccordionItem value={2} className="border-slate-800">
            <AccordionTrigger className="text-slate-200">
              Policy Citations
            </AccordionTrigger>
            <AccordionContent>
              <div className="flex flex-col gap-2">
                {data.policy_citations.map((citation) => {
                  const policy = policyById.get(citation.policy_id)
                  const similarity = policy?.similarity ?? 0
                  return (
                    <div
                      key={citation.policy_id}
                      className="flex flex-col gap-1.5 rounded-lg border border-slate-800 bg-slate-950/60 p-2.5"
                    >
                      <div className="flex items-center gap-2">
                        <Badge
                          variant="outline"
                          className="border-amber-500/40 font-mono text-[10px] text-amber-400"
                        >
                          {citation.policy_id}
                        </Badge>
                        {policy ? (
                          <span className="text-xs font-semibold text-slate-200">
                            {policy.title}
                          </span>
                        ) : null}
                      </div>
                      <p className="text-xs text-slate-400">{citation.why_relevant}</p>
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-800">
                          <div
                            className="h-full rounded-full bg-amber-500"
                            style={{ width: `${similarity * 100}%` }}
                          />
                        </div>
                        <span className="font-mono text-[11px] text-slate-400 tabular-nums">
                          {similarity.toFixed(4)}
                        </span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value={3} className="border-slate-800">
            <AccordionTrigger className="text-slate-200">
              Stream B — pgvector retrieval
            </AccordionTrigger>
            <AccordionContent>
              <div className="flex flex-col gap-2">
                {data.retrieved_policies.map((policy) => (
                  <PolicyBody key={policy.policy_id} policy={policy} />
                ))}
              </div>
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value={4} className="border-slate-800">
            <AccordionTrigger className="text-slate-200">
              Execution Trace
            </AccordionTrigger>
            <AccordionContent>
              <LatencyTable latency={data.latency_ms} />
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>
    </Shell>
  )
}

function LatencyRow({
  label,
  value,
  total,
}: {
  label: string
  value: number
  total?: boolean
}) {
  return (
    <div
      className={cn(
        "flex items-center justify-between py-1 font-mono text-xs tabular-nums",
        total
          ? "border-t border-slate-700 pt-1.5 font-semibold text-slate-100"
          : "text-slate-400",
      )}
    >
      <span className={cn(!total && "text-slate-500")}>{label}</span>
      <span>{formatMs(value)}</span>
    </div>
  )
}

function LatencyTable({
  latency,
}: {
  latency: UnderwriteResponse["latency_ms"]
}) {
  const total =
    latency.stream_a_ms +
    latency.stream_b_ms +
    latency.recourse_ms +
    latency.llm_ms +
    latency.verifier_ms
  return (
    <div className="flex flex-col">
      <LatencyRow label="Stream A" value={latency.stream_a_ms} />
      <LatencyRow label="Stream B" value={latency.stream_b_ms} />
      <LatencyRow label="Recourse" value={latency.recourse_ms} />
      <LatencyRow label="LLM" value={latency.llm_ms} />
      <LatencyRow label="Verifier" value={latency.verifier_ms} />
      <LatencyRow label="Total" value={total} total />
    </div>
  )
}
