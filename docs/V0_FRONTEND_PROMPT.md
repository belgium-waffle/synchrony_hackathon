# Deliverable 3 — v0.dev / Bolt.new Prompt

Copy everything between the horizontal rules into v0.dev or Bolt.new.

Two notes before you paste. First, generators drift on long prompts — if the output
is missing a panel, re-prompt with just that section rather than regenerating the
whole thing. Second, the exact API response shape is pasted into the prompt because
generators invent field names otherwise, and then nothing binds.

---

Build a single-page executive credit underwriting dashboard in React with TypeScript,
Tailwind CSS and shadcn/ui. This is an enterprise banking tool for Synchrony — it
should read as a serious internal risk console, not a consumer fintech app.

## Visual direction

- Dark slate background (`bg-slate-950`), panels as `bg-slate-900` cards with
  `border border-slate-800` and `rounded-xl`. No gradients, no glassmorphism.
- One accent colour only: amber-500 for primary actions and the score dial.
  Semantic colours are reserved for decisions: emerald-500 = APPROVE,
  amber-500 = REVIEW, rose-500 = DECLINE.
- Typography: `font-sans` for UI, `font-mono` for every number, policy ID and SHAP
  value. Numbers must never reflow when they change — use `tabular-nums`.
- Density over whitespace. This is a tool an underwriter uses for eight hours.

## Layout

A fixed top bar, then a three-column grid filling the viewport height:
`grid-cols-[360px_1fr_400px]` with `gap-4 p-4`. Each column scrolls independently
(`overflow-y-auto`). Below `lg`, stack to a single column in the order
left → center → right.

**Top bar:** left — "Next-Gen Credit Intelligence" with a small amber shield icon.
Right — three shadcn `Badge` components fed from `GET /health`: model version,
policy store (`pgvector` = emerald, `in-memory` = amber), LLM mode
(`bedrock` = emerald, `mock` = slate). Poll health once on mount.

### Left column — "Application Intake"

A shadcn `Card` containing a `Form` (react-hook-form + zod). Fields, in order:

| Field | Control | Notes |
|---|---|---|
| `sk_id_curr` | Input, number | Label "Applicant ID" |
| `amt_income_total` | Input, number | "Annual Income", prefix `$`, required, > 0 |
| `amt_credit` | Input, number | "Requested Credit", prefix `$`, required, > 0 |
| `amt_annuity` | Input, number | "Annuity (monthly)", prefix `$`, optional |
| `cnt_fam_members` | Input, number | "Household Size", min 1 |
| `days_birth` | Slider 20–69 | Label "Age", display in years, submit as `-(years * 365)` |
| `days_employed` | Input + Switch | Switch "Not currently employed" — when on, submit `365243` and disable the input. Show helper text: "Sends the Home Credit sentinel value." |
| `ext_source_1/2/3` | Three Sliders 0–1, step 0.01 | Group label "External Scores". Each has a "Not available" Checkbox that submits `null` and greys the slider. |
| `amt_req_credit_bureau_year` | Input, number | "Bureau Enquiries (12m)" |
| `def_30_cnt_social_circle` | Input, number | "Social Circle Defaults" |
| `name_education_type` | Select | Secondary / secondary special, Higher education, Incomplete higher, Lower secondary |
| `name_income_type` | Select | Working, Commercial associate, State servant, Pensioner |
| `occupation_type` | Select | Laborers, Sales staff, Core staff, Managers, Drivers |
| `underwriter_notes` | Textarea, 5 rows | **Required.** Label "Underwriter Notes". Helper text: "Drives policy retrieval. The scoring model never reads this field." |

Do **not** render inputs for gender, marital status or housing type — they are
excluded from the intake UI by design. Add a small muted footnote saying so.

Below the form: a full-width amber `Button` labelled "Run Dual-Stream Analysis",
disabled while loading, and a ghost `Button` "Load Sample Applicant" that fills the
form with a thin-file case (no employment, two external scores missing, 6 bureau
enquiries).

### Center column — "Risk Assessment"

**1. Score dial.** A semicircular gauge, 300 → 850, built with SVG (two `<path>`
arcs and `strokeDasharray`, no chart library). The arc colour follows the decision
band. Centered inside: the score in `text-6xl font-mono tabular-nums`, the decision
status beneath in a coloured `Badge`, and `PD 12.34%` in small muted mono text.
Animate the number counting up over 600ms on new results. Tick marks at 640 and 720
labelled "Review" and "Approve".

**2. SHAP attribution chart.** A Recharts horizontal `BarChart` from `key_factors`.
Title "TreeSHAP Feature Family Attribution", subtitle "Log-odds contribution;
positive values raise default risk". Requirements:
- `layout="vertical"`, `dataKey="family"` on the `YAxis` (width 170, no tick line),
  `contribution` on the `XAxis`.
- A `ReferenceLine x={0}` so bars diverge left and right from zero.
- Per-bar `Cell` colour: rose-500 when `direction === "increases_risk"`,
  emerald-500 when `decreases_risk`.
- `LabelList` showing `contribution_pct` as `44.6%` outside each bar end.
- Custom dark `Tooltip` showing family, exact contribution to 4 decimals, percentage,
  and the `top_features` array as a comma-separated mono list.
- Footer line under the chart: "Attribution computed with exact TreeSHAP, averaged
  across 5 fold models."

**3. Counterfactual recourse card.** Amber-bordered, lightbulb icon, heading
"Actionable Recourse". Render `recourse.advice` as the body text. When
`recourse.available` is true, also show a compact row: `display_name`,
`current_value → target_value`, and a `score_delta` pill (emerald if positive) reading
`+170 pts`. When `crosses_threshold` is non-null, add an emerald callout: "Would
reach {status}". When `available` is false, render the advice in muted text with no
pill.

### Right column — "Compliance Audit Trail"

Top of the column: a verification banner. When `verification.passed` is true, an
emerald bar with a shield-check icon reading "VERIFIED — {checks_run} deterministic
checks passed", then smaller muted text "{verifier_version}". This banner is the
single most important element on the page; make it prominent.

Then a shadcn `Accordion`, all sections open by default:

1. **Decision Summary** — `summary` as prose, plus a two-column mono key/value grid:
   Risk Score, PD, Status, Recommended Limit (currency formatted), Model Version.
2. **Adverse Action Reasons** — a list with rose bullet markers. Hide the whole
   section when the array is empty.
3. **Policy Citations** — one bordered row per `policy_citations` entry: the
   `policy_id` as an amber mono `Badge`, the matching title from `retrieved_policies`
   in semibold, `why_relevant` in muted text, and a similarity bar (a thin amber
   progress bar at `similarity * 100`%) with the value in mono.
4. **Retrieved Context** — each `retrieved_policies` entry with its full `body` in a
   `max-h-32 overflow-y-auto` block, `text-xs`. Label the section "Stream B — pgvector
   retrieval".
5. **Execution Trace** — the `latency_ms` object as a mono table: Stream A, Stream B,
   Recourse, LLM, Verifier, each in ms, plus a total row.

## API integration

Base URL from `process.env.NEXT_PUBLIC_API_URL`, default `http://localhost:8000`.

`POST {base}/api/v1/underwrite` with `Content-Type: application/json`. The request
body is the flat snake_case object described in the form table. The response is
exactly this shape — bind to these names, do not rename them:

```json
{
  "applicant_id": "100012",
  "request_id": "36a3b1c5-...",
  "model_version": "stream-a-lgbm-aguiar-1.0.0",
  "generated_at": "2026-01-01T00:00:00Z",
  "risk_score": 349,
  "probability_of_default": 0.910759,
  "approval_status": "DECLINE",
  "recommended_credit_limit": 0.0,
  "summary": "Applicant 100012 scores 349 ...",
  "key_factors": [
    { "family": "External Scores", "contribution": 2.7731,
      "contribution_pct": 70.3, "direction": "increases_risk",
      "top_features": ["EXT_SOURCE_3", "EXT_SOURCE_2"] }
  ],
  "policy_citations": [
    { "policy_id": "UW-3.2", "why_relevant": "..." }
  ],
  "retrieved_policies": [
    { "policy_id": "UW-3.2", "title": "Reliance on external scoring sources",
      "body": "...", "similarity": 0.5827 }
  ],
  "adverse_action_reasons": ["External Scores weighed against this application."],
  "recourse": {
    "available": true, "feature": "EXT_SOURCE_3",
    "display_name": "External score 3 (currently missing)",
    "current_value": null, "target_value": 0.55,
    "projected_score": 519, "score_delta": 170,
    "crosses_threshold": null, "horizon_months": 6,
    "advice": "Actionable Advice: if you ..."
  },
  "verification": { "passed": true, "checks_run": 20, "failures": [],
                    "verifier_version": "deterministic-verifier-1.2.0" },
  "latency_ms": { "stream_a_ms": 22.4, "stream_b_ms": 0.1, "recourse_ms": 12.3,
                  "llm_ms": 0.1, "verifier_ms": 0.3 }
}
```

### Loading state

The dual-stream call takes a few seconds, so do not show one spinner. Show a staged
progress list in the center column that advances on a timer while the request is in
flight: "Stream A — scoring and TreeSHAP" → "Stream B — retrieving policy from
pgvector" → "Synthesising audit narrative" → "Running deterministic verification".
Completed stages get an emerald check, the active stage a spinner, pending stages a
muted dot. Skeleton-load the left and right panels with shadcn `Skeleton`.

### Error handling — the important case

A `422` response means **the verifier blocked the decision**. This is a feature, not
a crash, and it must look deliberate. The response body is:

```json
{ "detail": { "error": "verification_failed",
              "message": "...", "failures": ["..."],
              "verifier_version": "...", "risk_score": 349 } }
```

Render a rose-bordered `Alert` filling the right column: heading "DECISION BLOCKED BY
VERIFIER", the `message`, then every entry in `failures` as a mono list item with a
rose `AlertTriangle` icon. Add the line: "No decision was released. The generated
narrative failed deterministic validation against the model output." Do not show a
score dial or SHAP chart in this state — show an empty state in the center instead.

Handle `502` (Bedrock unavailable) and network failure with a plain amber alert and a
"Retry" button. Never render a partial or cached report after an error.

Also add a small ghost `Button` in the top bar labelled "Demo: Verifier" that calls
`POST {base}/api/v1/underwrite/verify-demo` with the current form values and displays
the returned `failures` in the same rose alert. This is the negative-control demo.

## Engineering requirements

- TypeScript interfaces for every API shape, in `types/api.ts`. No `any`.
- One `useUnderwrite()` hook owning `loading`, `data`, `error` and the POST. Components
  stay presentational.
- Currency via `Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD',
  maximumFractionDigits: 0 })`.
- Every SHAP number rendered to exactly 4 decimals; never round for display.
- `aria-live="polite"` on the verification banner and score dial.
- No mock data anywhere in the committed component. Empty state before the first
  submit reads "Submit an application to run the dual-stream analysis."

---
