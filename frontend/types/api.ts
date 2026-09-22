export type ApprovalStatus = "APPROVE" | "REVIEW" | "DECLINE"

export type ShapDirection = "increases_risk" | "decreases_risk"

export interface KeyFactor {
  family: string
  contribution: number
  contribution_pct: number
  direction: ShapDirection
  top_features: string[]
}

export interface PolicyCitation {
  policy_id: string
  why_relevant: string
}

export interface RetrievedPolicy {
  policy_id: string
  title: string
  body: string
  similarity: number
}

export interface Recourse {
  available: boolean
  feature: string | null
  display_name: string | null
  current_value: number | null
  target_value: number | null
  projected_score: number | null
  score_delta: number | null
  crosses_threshold: string | null
  horizon_months: number | null
  advice: string
}

export interface Verification {
  passed: boolean
  checks_run: number
  failures: string[]
  verifier_version: string
}

export interface LatencyMs {
  stream_a_ms: number
  stream_b_ms: number
  recourse_ms: number
  llm_ms: number
  verifier_ms: number
}

export interface UnderwriteResponse {
  applicant_id: string
  request_id: string
  model_version: string
  generated_at: string
  risk_score: number
  probability_of_default: number
  approval_status: ApprovalStatus
  recommended_credit_limit: number
  summary: string
  key_factors: KeyFactor[]
  policy_citations: PolicyCitation[]
  retrieved_policies: RetrievedPolicy[]
  adverse_action_reasons: string[]
  recourse: Recourse
  verification: Verification
  latency_ms: LatencyMs
}

export interface VerifierFailureDetail {
  error: "verification_failed"
  message: string
  failures: string[]
  verifier_version: string
  risk_score: number
}

export interface VerifierBlockedResponse {
  detail: VerifierFailureDetail
}

export interface HealthResponse {
  model_version: string
  policy_store: "pgvector" | "in-memory"
  llm_mode: "bedrock" | "mock"
}

export interface UnderwriteRequest {
  sk_id_curr: number | null
  amt_income_total: number
  amt_credit: number
  amt_annuity: number | null
  cnt_fam_members: number
  days_birth: number
  days_employed: number
  ext_source_1: number | null
  ext_source_2: number | null
  ext_source_3: number | null
  amt_req_credit_bureau_year: number | null
  def_30_cnt_social_circle: number | null
  name_education_type: string
  name_income_type: string
  occupation_type: string
  underwriter_notes: string
}

export type UnderwriteErrorKind = "verifier_blocked" | "bedrock" | "network"

export interface UnderwriteError {
  kind: UnderwriteErrorKind
  message: string
  detail?: VerifierFailureDetail
}
