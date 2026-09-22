"use client"

import { Activity, ShieldAlert } from "lucide-react"

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { LoadingStages } from "@/components/dashboard/loading-stages"
import { RecourseCard } from "@/components/dashboard/recourse-card"
import { ScoreDial } from "@/components/dashboard/score-dial"
import { ShapChart } from "@/components/dashboard/shap-chart"
import type { UnderwriteError, UnderwriteResponse } from "@/types/api"

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <Card className="gap-0 border-slate-800 bg-slate-900 py-0">
      <CardHeader className="border-b border-slate-800 px-4 py-3">
        <CardTitle className="text-sm font-semibold tracking-wide text-slate-100 uppercase">
          Risk Assessment
        </CardTitle>
      </CardHeader>
      <CardContent className="px-4 py-5">{children}</CardContent>
    </Card>
  )
}

export function CenterColumn({
  loading,
  data,
  error,
  hasSubmitted,
}: {
  loading: boolean
  data: UnderwriteResponse | null
  error: UnderwriteError | null
  hasSubmitted: boolean
}) {
  if (loading) {
    return (
      <Shell>
        <LoadingStages />
      </Shell>
    )
  }

  // When the verifier blocks the decision, the center shows an empty state and
  // the rose alert lives in the right column.
  if (error?.kind === "verifier_blocked") {
    return (
      <Shell>
        <div className="flex min-h-64 flex-col items-center justify-center gap-3 text-center">
          <ShieldAlert className="size-8 text-rose-500/70" aria-hidden />
          <p className="max-w-xs text-sm text-slate-400">
            No decision was released. See the verifier report in the compliance
            column.
          </p>
        </div>
      </Shell>
    )
  }

  if (!data || !hasSubmitted) {
    return (
      <Shell>
        <div className="flex min-h-64 flex-col items-center justify-center gap-3 text-center">
          <Activity className="size-8 text-slate-700" aria-hidden />
          <p className="max-w-xs text-sm text-slate-500">
            Submit an application to run the dual-stream analysis.
          </p>
        </div>
      </Shell>
    )
  }

  return (
    <Shell>
      <div className="flex flex-col gap-6">
        <ScoreDial
          score={data.risk_score}
          status={data.approval_status}
          pd={data.probability_of_default}
        />
        <Separator className="bg-slate-800" />
        <ShapChart factors={data.key_factors} />
        <Separator className="bg-slate-800" />
        <RecourseCard recourse={data.recourse} />
      </div>
    </Shell>
  )
}
