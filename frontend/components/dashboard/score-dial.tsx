"use client"

import { useCountUp } from "@/hooks/use-count-up"
import {
  APPROVE_THRESHOLD,
  REVIEW_THRESHOLD,
  SCORE_MAX,
  SCORE_MIN,
  getDecisionMeta,
  scoreFraction,
} from "@/lib/decision"
import { formatPd } from "@/lib/format"
import type { ApprovalStatus } from "@/types/api"
import { cn } from "@/lib/utils"

const CX = 120
const CY = 120
const R = 100

function polar(score: number) {
  const f = scoreFraction(score)
  const angle = (180 - f * 180) * (Math.PI / 180)
  return {
    x: CX + R * Math.cos(angle),
    y: CY - R * Math.sin(angle),
  }
}

function Tick({ score, label }: { score: number; label: string }) {
  const inner = { ...polar(score) }
  const f = scoreFraction(score)
  const angle = (180 - f * 180) * (Math.PI / 180)
  const outerR = R + 12
  const ox = CX + outerR * Math.cos(angle)
  const oy = CY - outerR * Math.sin(angle)
  const lx = CX + (outerR + 4) * Math.cos(angle)
  const ly = CY - (outerR + 4) * Math.sin(angle)
  return (
    <g>
      <line
        x1={inner.x}
        y1={inner.y}
        x2={ox}
        y2={oy}
        stroke="#475569"
        strokeWidth={1.5}
      />
      <text
        x={lx}
        y={ly}
        fill="#64748b"
        fontSize={9}
        textAnchor={lx < CX ? "end" : "start"}
        dominantBaseline="middle"
        className="font-mono"
      >
        {label}
      </text>
    </g>
  )
}

export function ScoreDial({
  score,
  status,
  pd,
}: {
  score: number
  status: ApprovalStatus
  pd: number
}) {
  const meta = getDecisionMeta(status)
  const animated = useCountUp(score, 600)
  const fraction = scoreFraction(score)

  return (
    <div
      className="flex flex-col items-center"
      aria-live="polite"
      aria-label={`Risk score ${score}, decision ${meta.label}, probability of default ${formatPd(pd)}`}
    >
      <div className="relative w-full max-w-[280px]">
        <svg viewBox="0 0 240 150" className="w-full">
          <path
            d={`M ${CX - R} ${CY} A ${R} ${R} 0 0 1 ${CX + R} ${CY}`}
            fill="none"
            stroke="#1e293b"
            strokeWidth={14}
            strokeLinecap="round"
            pathLength={100}
          />
          <path
            d={`M ${CX - R} ${CY} A ${R} ${R} 0 0 1 ${CX + R} ${CY}`}
            fill="none"
            stroke={meta.hex}
            strokeWidth={14}
            strokeLinecap="round"
            pathLength={100}
            strokeDasharray={`${fraction * 100} 100`}
            style={{ transition: "stroke-dasharray 600ms ease-out" }}
          />
          <Tick score={REVIEW_THRESHOLD} label="Review" />
          <Tick score={APPROVE_THRESHOLD} label="Approve" />
          <text
            x={CX - R}
            y={CY + 16}
            fill="#475569"
            fontSize={9}
            textAnchor="middle"
            className="font-mono"
          >
            {SCORE_MIN}
          </text>
          <text
            x={CX + R}
            y={CY + 16}
            fill="#475569"
            fontSize={9}
            textAnchor="middle"
            className="font-mono"
          >
            {SCORE_MAX}
          </text>
        </svg>

        <div className="absolute inset-x-0 bottom-1 flex flex-col items-center gap-1">
          <span
            className={cn(
              "font-mono text-6xl leading-none font-bold tabular-nums",
              meta.text,
            )}
          >
            {animated}
          </span>
          <span
            className={cn(
              "rounded-md border px-2.5 py-0.5 font-mono text-xs font-semibold tracking-wide",
              meta.text,
              meta.border,
              meta.bg,
            )}
          >
            {meta.label}
          </span>
          <span className="font-mono text-xs text-slate-500 tabular-nums">
            PD {formatPd(pd)}
          </span>
        </div>
      </div>
    </div>
  )
}
