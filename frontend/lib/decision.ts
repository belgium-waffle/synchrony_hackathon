import type { ApprovalStatus } from "@/types/api"

export interface DecisionMeta {
  label: string
  hex: string
  /** tailwind text colour */
  text: string
  /** tailwind border colour */
  border: string
  /** tailwind background tint */
  bg: string
}

export const decisionMeta: Record<ApprovalStatus, DecisionMeta> = {
  APPROVE: {
    label: "APPROVE",
    hex: "#10b981",
    text: "text-emerald-400",
    border: "border-emerald-500/50",
    bg: "bg-emerald-500/10",
  },
  REVIEW: {
    label: "REVIEW",
    hex: "#f59e0b",
    text: "text-amber-400",
    border: "border-amber-500/50",
    bg: "bg-amber-500/10",
  },
  DECLINE: {
    label: "DECLINE",
    hex: "#f43f5e",
    text: "text-rose-400",
    border: "border-rose-500/50",
    bg: "bg-rose-500/10",
  },
}

export function getDecisionMeta(status: ApprovalStatus): DecisionMeta {
  return decisionMeta[status] ?? decisionMeta.REVIEW
}

export const SCORE_MIN = 300
export const SCORE_MAX = 850
export const REVIEW_THRESHOLD = 640
export const APPROVE_THRESHOLD = 720

export function scoreFraction(score: number): number {
  const clamped = Math.min(SCORE_MAX, Math.max(SCORE_MIN, score))
  return (clamped - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)
}
