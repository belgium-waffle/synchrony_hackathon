"use client"

import { Controller, type UseFormReturn } from "react-hook-form"
import { Play, FileDown } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import {
  sampleApplicant,
  type UnderwriteFormValues,
} from "@/lib/underwrite-form"
import { cn } from "@/lib/utils"

const labelCls = "text-xs font-medium text-slate-300"
const helperCls = "text-[11px] leading-snug text-slate-500"
const errorCls = "text-[11px] font-medium text-rose-400"
const inputCls =
  "h-8 border-slate-700 bg-slate-950 font-mono text-sm text-slate-100 tabular-nums placeholder:text-slate-600"

const EDUCATION_OPTIONS = [
  "Secondary / secondary special",
  "Higher education",
  "Incomplete higher",
  "Lower secondary",
]
const INCOME_OPTIONS = [
  "Working",
  "Commercial associate",
  "State servant",
  "Pensioner",
]
const OCCUPATION_OPTIONS = [
  "Laborers",
  "Sales staff",
  "Core staff",
  "Managers",
  "Drivers",
]

function Field({
  label,
  htmlFor,
  children,
  helper,
  error,
}: {
  label: string
  htmlFor?: string
  children: React.ReactNode
  helper?: string
  error?: string
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={htmlFor} className={labelCls}>
        {label}
      </label>
      {children}
      {helper ? <p className={helperCls}>{helper}</p> : null}
      {error ? <p className={errorCls}>{error}</p> : null}
    </div>
  )
}

function MoneyInput({
  id,
  register,
  placeholder,
}: {
  id: string
  register: ReturnType<UseFormReturn<UnderwriteFormValues>["register"]>
  placeholder?: string
}) {
  return (
    <div className="relative">
      <span className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 font-mono text-sm text-slate-500">
        $
      </span>
      <Input
        id={id}
        type="number"
        inputMode="decimal"
        placeholder={placeholder}
        className={cn(inputCls, "pl-6")}
        {...register}
      />
    </div>
  )
}

function ExternalScore({
  index,
  form,
}: {
  index: 1 | 2 | 3
  form: UseFormReturn<UnderwriteFormValues>
}) {
  const scoreKey = `ext_source_${index}` as const
  const naKey = `ext_source_${index}_na` as const
  const na = form.watch(naKey)
  const value = form.watch(scoreKey)

  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-slate-800 bg-slate-950/60 p-2.5">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-slate-300">
          External Score {index}
        </span>
        <span
          className={cn(
            "font-mono text-xs tabular-nums",
            na ? "text-slate-600" : "text-slate-200",
          )}
        >
          {na ? "n/a" : value.toFixed(2)}
        </span>
      </div>
      <Controller
        control={form.control}
        name={scoreKey}
        render={({ field }) => (
          <Slider
            min={0}
            max={1}
            step={0.01}
            value={field.value}
            onValueChange={(v) =>
              field.onChange(Array.isArray(v) ? v[0] : v)
            }
            disabled={na}
            className={cn(na && "opacity-40")}
          />
        )}
      />
      <label className="flex items-center gap-2 text-[11px] text-slate-400">
        <Controller
          control={form.control}
          name={naKey}
          render={({ field }) => (
            <Checkbox
              checked={field.value}
              onCheckedChange={(checked) => field.onChange(Boolean(checked))}
            />
          )}
        />
        Not available (submits null)
      </label>
    </div>
  )
}

export function IntakeForm({
  form,
  onSubmit,
  loading,
}: {
  form: UseFormReturn<UnderwriteFormValues>
  onSubmit: (values: UnderwriteFormValues) => void
  loading: boolean
}) {
  const {
    register,
    control,
    handleSubmit,
    reset,
    watch,
    formState: { errors },
  } = form

  const notEmployed = watch("not_employed")
  const age = watch("age")

  return (
    <Card className="gap-0 border-slate-800 bg-slate-900 py-0">
      <CardHeader className="gap-1 border-b border-slate-800 px-4 py-3">
        <CardTitle className="text-sm font-semibold tracking-wide text-slate-100 uppercase">
          Application Intake
        </CardTitle>
        <CardDescription className="text-[11px] text-slate-500">
          Fields feed Stream A scoring and Stream B policy retrieval.
        </CardDescription>
      </CardHeader>
      <CardContent className="px-4 py-4">
        <form
          onSubmit={handleSubmit(onSubmit)}
          className="flex flex-col gap-4"
          noValidate
        >
          <Field label="Applicant ID" htmlFor="sk_id_curr">
            <Input
              id="sk_id_curr"
              type="number"
              placeholder="Optional"
              className={inputCls}
              {...register("sk_id_curr")}
            />
          </Field>

          <Field
            label="Annual Income"
            htmlFor="amt_income_total"
            error={errors.amt_income_total?.message}
          >
            <MoneyInput
              id="amt_income_total"
              register={register("amt_income_total")}
              placeholder="0"
            />
          </Field>

          <Field
            label="Requested Credit"
            htmlFor="amt_credit"
            error={errors.amt_credit?.message}
          >
            <MoneyInput
              id="amt_credit"
              register={register("amt_credit")}
              placeholder="0"
            />
          </Field>

          <Field label="Annuity (monthly)" htmlFor="amt_annuity">
            <MoneyInput
              id="amt_annuity"
              register={register("amt_annuity")}
              placeholder="Optional"
            />
          </Field>

          <Field
            label="Household Size"
            htmlFor="cnt_fam_members"
            error={errors.cnt_fam_members?.message}
          >
            <Input
              id="cnt_fam_members"
              type="number"
              min={1}
              className={inputCls}
              {...register("cnt_fam_members")}
            />
          </Field>

          <Field label={`Age — ${age} years`}>
            <Controller
              control={control}
              name="age"
              render={({ field }) => (
                <Slider
                  min={20}
                  max={69}
                  step={1}
                  value={field.value}
                  onValueChange={(v) =>
                    field.onChange(Array.isArray(v) ? v[0] : v)
                  }
                />
              )}
            />
          </Field>

          <Field
            label="Employment"
            htmlFor="days_employed"
            helper="Sends the Home Credit sentinel value (365243) when not employed."
          >
            <div className="flex items-center gap-2">
              <Input
                id="days_employed"
                type="number"
                disabled={notEmployed}
                className={cn(inputCls, "flex-1")}
                {...register("days_employed")}
              />
            </div>
            <label className="mt-1 flex items-center gap-2 text-[11px] text-slate-400">
              <Controller
                control={control}
                name="not_employed"
                render={({ field }) => (
                  <Switch
                    checked={field.value}
                    onCheckedChange={(checked) => field.onChange(Boolean(checked))}
                  />
                )}
              />
              Not currently employed
            </label>
          </Field>

          <div className="flex flex-col gap-2">
            <span className={labelCls}>External Scores</span>
            <ExternalScore index={1} form={form} />
            <ExternalScore index={2} form={form} />
            <ExternalScore index={3} form={form} />
          </div>

          <Field label="Bureau Enquiries (12m)" htmlFor="amt_req_credit_bureau_year">
            <Input
              id="amt_req_credit_bureau_year"
              type="number"
              placeholder="Optional"
              className={inputCls}
              {...register("amt_req_credit_bureau_year")}
            />
          </Field>

          <Field label="Social Circle Defaults" htmlFor="def_30_cnt_social_circle">
            <Input
              id="def_30_cnt_social_circle"
              type="number"
              placeholder="Optional"
              className={inputCls}
              {...register("def_30_cnt_social_circle")}
            />
          </Field>

          <SelectField
            label="Education"
            control={control}
            name="name_education_type"
            options={EDUCATION_OPTIONS}
            error={errors.name_education_type?.message}
          />
          <SelectField
            label="Income Type"
            control={control}
            name="name_income_type"
            options={INCOME_OPTIONS}
            error={errors.name_income_type?.message}
          />
          <SelectField
            label="Occupation"
            control={control}
            name="occupation_type"
            options={OCCUPATION_OPTIONS}
            error={errors.occupation_type?.message}
          />

          <Field
            label="Underwriter Notes"
            htmlFor="underwriter_notes"
            helper="Drives policy retrieval. The scoring model never reads this field."
            error={errors.underwriter_notes?.message}
          >
            <Textarea
              id="underwriter_notes"
              rows={5}
              className="resize-none border-slate-700 bg-slate-950 text-sm text-slate-100 placeholder:text-slate-600"
              placeholder="Describe the case, risk considerations, and policy context…"
              {...register("underwriter_notes")}
            />
          </Field>

          <p className="text-[11px] leading-snug text-slate-600">
            Gender, marital status and housing type are intentionally excluded
            from the intake UI by design.
          </p>

          <div className="flex flex-col gap-2 pt-1">
            <Button
              type="submit"
              disabled={loading}
              className="w-full bg-amber-500 font-semibold text-slate-950 hover:bg-amber-400"
            >
              <Play data-icon="inline-start" />
              {loading ? "Analysing…" : "Run Dual-Stream Analysis"}
            </Button>
            <Button
              type="button"
              variant="ghost"
              disabled={loading}
              onClick={() => reset(sampleApplicant)}
              className="w-full text-slate-300 hover:bg-slate-800 hover:text-slate-100"
            >
              <FileDown data-icon="inline-start" />
              Load Sample Applicant
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  )
}

function SelectField({
  label,
  control,
  name,
  options,
  error,
}: {
  label: string
  control: UseFormReturn<UnderwriteFormValues>["control"]
  name: "name_education_type" | "name_income_type" | "occupation_type"
  options: string[]
  error?: string
}) {
  return (
    <Field label={label} error={error}>
      <Controller
        control={control}
        name={name}
        render={({ field }) => (
          <Select
            value={field.value}
            onValueChange={(v) => field.onChange(v as string)}
          >
            <SelectTrigger className="h-8 w-full border-slate-700 bg-slate-950 text-sm text-slate-100">
              <SelectValue placeholder="Select…" />
            </SelectTrigger>
            <SelectContent className="border-slate-700 bg-slate-900 text-slate-100">
              <SelectGroup>
                {options.map((opt) => (
                  <SelectItem key={opt} value={opt}>
                    {opt}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        )}
      />
    </Field>
  )
}
