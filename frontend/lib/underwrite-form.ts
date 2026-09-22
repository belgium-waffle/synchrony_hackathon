import { z } from "zod"

import type { UnderwriteRequest } from "@/types/api"

const positiveMoney = (label: string) =>
  z
    .string()
    .min(1, `${label} is required`)
    .refine((v) => Number(v) > 0, `${label} must be greater than 0`)

export const underwriteFormSchema = z.object({
  sk_id_curr: z.string(),
  amt_income_total: positiveMoney("Annual Income"),
  amt_credit: positiveMoney("Requested Credit"),
  amt_annuity: z.string(),
  cnt_fam_members: z
    .string()
    .refine((v) => v === "" || Number(v) >= 1, "Household size must be at least 1"),
  age: z.number().min(20).max(69),
  not_employed: z.boolean(),
  days_employed: z.string(),
  ext_source_1: z.number().min(0).max(1),
  ext_source_1_na: z.boolean(),
  ext_source_2: z.number().min(0).max(1),
  ext_source_2_na: z.boolean(),
  ext_source_3: z.number().min(0).max(1),
  ext_source_3_na: z.boolean(),
  amt_req_credit_bureau_year: z.string(),
  def_30_cnt_social_circle: z.string(),
  name_education_type: z.string().min(1, "Select an education type"),
  name_income_type: z.string().min(1, "Select an income type"),
  occupation_type: z.string().min(1, "Select an occupation type"),
  underwriter_notes: z.string().min(1, "Underwriter notes are required"),
})

export type UnderwriteFormValues = z.infer<typeof underwriteFormSchema>

export const DAYS_EMPLOYED_SENTINEL = 365243

export const defaultFormValues: UnderwriteFormValues = {
  sk_id_curr: "",
  amt_income_total: "",
  amt_credit: "",
  amt_annuity: "",
  cnt_fam_members: "1",
  age: 40,
  not_employed: false,
  days_employed: "-1500",
  ext_source_1: 0.5,
  ext_source_1_na: false,
  ext_source_2: 0.5,
  ext_source_2_na: false,
  ext_source_3: 0.5,
  ext_source_3_na: false,
  amt_req_credit_bureau_year: "",
  def_30_cnt_social_circle: "",
  name_education_type: "",
  name_income_type: "",
  occupation_type: "",
  underwriter_notes: "",
}

// A thin-file case: not currently employed, two external scores missing,
// six bureau enquiries.
export const sampleApplicant: UnderwriteFormValues = {
  sk_id_curr: "100012",
  amt_income_total: "112500",
  amt_credit: "450000",
  amt_annuity: "24700",
  cnt_fam_members: "3",
  age: 52,
  not_employed: true,
  days_employed: String(DAYS_EMPLOYED_SENTINEL),
  ext_source_1: 0.5,
  ext_source_1_na: true,
  ext_source_2: 0.34,
  ext_source_2_na: false,
  ext_source_3: 0.5,
  ext_source_3_na: true,
  amt_req_credit_bureau_year: "6",
  def_30_cnt_social_circle: "1",
  name_education_type: "Secondary / secondary special",
  name_income_type: "Working",
  occupation_type: "Laborers",
  underwriter_notes:
    "Thin-file applicant, recently unemployed. Two of three external bureau scores are unavailable and there have been six credit bureau enquiries in the last twelve months. Requesting review against reliance-on-external-scoring policy.",
}

function optionalNumber(raw: string): number | null {
  if (raw.trim() === "") return null
  const n = Number(raw)
  return Number.isFinite(n) ? n : null
}

export function toUnderwriteRequest(
  values: UnderwriteFormValues,
): UnderwriteRequest {
  return {
    sk_id_curr: optionalNumber(values.sk_id_curr),
    amt_income_total: Number(values.amt_income_total),
    amt_credit: Number(values.amt_credit),
    amt_annuity: optionalNumber(values.amt_annuity),
    cnt_fam_members: values.cnt_fam_members ? Number(values.cnt_fam_members) : 1,
    days_birth: -(values.age * 365),
    days_employed: values.not_employed
      ? DAYS_EMPLOYED_SENTINEL
      : Number(values.days_employed || 0),
    ext_source_1: values.ext_source_1_na ? null : values.ext_source_1,
    ext_source_2: values.ext_source_2_na ? null : values.ext_source_2,
    ext_source_3: values.ext_source_3_na ? null : values.ext_source_3,
    amt_req_credit_bureau_year: optionalNumber(values.amt_req_credit_bureau_year),
    def_30_cnt_social_circle: optionalNumber(values.def_30_cnt_social_circle),
    name_education_type: values.name_education_type,
    name_income_type: values.name_income_type,
    occupation_type: values.occupation_type,
    underwriter_notes: values.underwriter_notes,
  }
}
