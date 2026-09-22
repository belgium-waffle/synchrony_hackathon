"use client"

import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import { formatPercent, formatShap } from "@/lib/format"
import type { KeyFactor } from "@/types/api"

const RISE = "#f43f5e" // rose-500
const FALL = "#10b981" // emerald-500

function ShapTooltip({
  active,
  payload,
}: {
  active?: boolean
  payload?: Array<{ payload: KeyFactor }>
}) {
  if (!active || !payload || payload.length === 0) return null
  const f = payload[0].payload
  const rising = f.direction === "increases_risk"
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-950 p-3 text-xs shadow-xl">
      <p className="mb-1 font-semibold text-slate-100">{f.family}</p>
      <dl className="flex flex-col gap-0.5 font-mono tabular-nums">
        <div className="flex justify-between gap-6">
          <dt className="text-slate-500">contribution</dt>
          <dd style={{ color: rising ? RISE : FALL }}>
            {formatShap(f.contribution)}
          </dd>
        </div>
        <div className="flex justify-between gap-6">
          <dt className="text-slate-500">share</dt>
          <dd className="text-slate-300">{formatPercent(f.contribution_pct)}</dd>
        </div>
        <div className="flex justify-between gap-6">
          <dt className="text-slate-500">direction</dt>
          <dd className="text-slate-300">
            {rising ? "increases risk" : "decreases risk"}
          </dd>
        </div>
      </dl>
      <p className="mt-1.5 border-t border-slate-800 pt-1.5 font-mono text-[11px] text-slate-400">
        {f.top_features.join(", ")}
      </p>
    </div>
  )
}

export function ShapChart({ factors }: { factors: KeyFactor[] }) {
  const data = [...(factors || [])].sort((a, b) => a.contribution - b.contribution)
  const height = Math.max(180, data.length * 46)

  return (
    <div className="flex flex-col gap-2">
      <div>
        <h3 className="text-sm font-semibold text-slate-100">
          TreeSHAP Feature Family Attribution
        </h3>
        <p className="text-[11px] text-slate-500">
          Log-odds contribution; positive values raise default risk.
        </p>
      </div>
      <div style={{ width: "100%", height }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 4, right: 56, bottom: 4, left: 4 }}
          >
            <XAxis
              type="number"
              dataKey="contribution"
              stroke="#475569"
              tick={{ fill: "#64748b", fontSize: 10 }}
              tickFormatter={(v: number) => v.toFixed(2)}
            />
            <YAxis
              type="category"
              dataKey="family"
              width={170}
              tickLine={false}
              axisLine={false}
              tick={{ fill: "#cbd5e1", fontSize: 11 }}
            />
            <ReferenceLine x={0} stroke="#64748b" strokeWidth={1} />
            <Tooltip
              cursor={{ fill: "rgba(148,163,184,0.08)" }}
              content={<ShapTooltip />}
            />
            <Bar dataKey="contribution" radius={2} isAnimationActive={false}>
              {data.map((f) => (
                <Cell
                  key={f.family}
                  fill={f.direction === "increases_risk" ? RISE : FALL}
                />
              ))}
              <LabelList
                dataKey="contribution_pct"
                position="right"
                formatter={(v: number) => formatPercent(v)}
                className="fill-slate-400 font-mono"
                fontSize={10}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-[11px] text-slate-500">
        Attribution computed with exact TreeSHAP, averaged across 5 fold models.
      </p>
    </div>
  )
}
