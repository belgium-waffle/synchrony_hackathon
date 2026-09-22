# Next-Gen Credit Intelligence

**A real-time, multi-modal underwriting engine for New-to-Credit (NTC) and thin-file applicants.**

Built for the Synchrony Hackathon — Problem Statement 1: *expand credit access using alternative data and real-time behavioral signals, while mitigating fraud and ensuring regulatory transparency.*

📄 Full technical report: [System Design Document (PDF)](synxhro.pdf)

---

## Why this exists

Traditional credit models lean on static bureau history, which locks out millions of otherwise creditworthy people who simply don't have a formal banking footprint. This project reframes the problem as **two problems, not one**:

1. **Quantitative** — score risk accurately from thin, alternative, and behavioral data.
2. **Qualitative / regulatory** — explain *why* a decision was made, in language a compliance officer and an applicant can both trust, without letting a language model invent the reasoning.

We solve both with a **Dual-Stream Architecture** that keeps deterministic ML and generative explanation strictly separated — reunited only behind a verifier that refuses to release anything it cannot mathematically prove.

```
[Input: Applicant Dictionary (Tabular + Notes)]
                               │
       ┌───────────────────────┴───────────────────────┐
       │                                               │
       ▼                                               ▼
┌───────────────────────────────┐        ┌───────────────────────────────┐
│   Stream A: ML Scoring Engine │        │   Stream B: Qualitative RAG   │
│                               │        │                               │
│  1. Preprocessing (NaN handle)│        │  1. Embed Applicant Notes     │
│  2. CatBoost + LightGBM Stack │        │  2. sentence-transformers     │
│  3. SHAP Feature Attribution  │        │  3. FAISS (Local Vector DB)   │
└───────────────┬───────────────┘        └───────────────┬───────────────┘
                │                                        │
                │ Risk Score + SHAP Values               │ Matched Policy Rules
                └───────────────────┬────────────────────┘
                                    │
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │              LLM Synthesis & Guardrail Layer           │
        │       (boto3 AWS Bedrock -> Strict JSON Output)        │
        └───────────────────────────┬────────────────────────────┘
                                    │ Structured Decision Payload
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │                 Deterministic Verifier                 │
        │   (Asserts LLM JSON matches exact SHAP & Policy IDs)   │
        └────────────────────────────────────────────────────────┘
```

If the LLM hallucinates a number or a policy citation, the system **fails closed** — it returns a blocked `422` decision, never a plausible-looking but false one.

---

## Key features

- **Mathematical, not narrative, explainability.** The LLM is handed exact SHAP feature-family attributions and must cite them verbatim; it never invents a reason.
- **Fails closed.** Any hallucinated policy ID, altered SHAP number, or misaligned reasoning blocks the response instead of degrading silently.
- **Alternative data is first-class.** Thin-file signals (utility payments, telecom recency, missingness counts) feed the scoring model directly.
- **Responsible AI by construction.** Protected attributes (gender, marital status, housing type) are excluded from the intake schema entirely, not down-weighted.
- **Counterfactual recourse.** Every decline comes with a concrete, re-scored statement of what would change the outcome.
- **Swappable LLM backend.** Written against a Bedrock-shaped `LLMClient` interface; a real `boto3` Bedrock client is a drop-in replacement for the current Gemini implementation.
- **Strong offline performance.** The blended LightGBM + CatBoost scoring engine achieves **0.7962 AUC** on out-of-fold validation.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React (TypeScript), Tailwind CSS, shadcn/ui |
| Backend / API | FastAPI (async, OpenAPI/Swagger-documented) |
| Database | PostgreSQL + `pgvector`, plus a JSONB audit-log table |
| LLM | Google Gemini (native SDK) behind a Bedrock-shaped `LLMClient` interface |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| ML models | LightGBM + CatBoost, blended in margin space; exact TreeSHAP |
| Model persistence | `joblib` artifact, trained once from the enriched dataset |

### Dataset

We train and evaluate on Kaggle's **[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)** dataset — the closest large, real-world analogue to the NTC/thin-file problem, with genuine multi-table behavioral signal (bureau history, previous applications, POS/credit-card balances, installment payments) and naturally occurring missingness.

The raw Kaggle CSVs are **not preprocessed out of the box** — they need to be joined and feature-engineered before the scoring engine can train on them. `backend/scripts/build_kaggle_features.py` handles this: it aggregates `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`, and `installments_payments.csv` to the applicant grain (`SK_ID_CURR`), applies the Active/Closed and Approved/Refused split tricks, engineers ratio and PCA features on `EXT_SOURCE_1/2/3`, and joins everything onto `application_train.csv` to produce `data/application_train_enriched.csv`. This is a **one-time offline job** — anyone setting up the repo needs to run it once against the raw Kaggle CSVs before the backend can train the model (see [Getting started](#getting-started) below).

---

## Repository structure

```
synchrony_credit_intelligence/
├── backend/
│   ├── app/
│   │   ├── bedrock.py        # Swappable LLM client (Gemini / Template / Mock)
│   │   ├── config.py         # Environment-driven settings
│   │   ├── database.py       # PostgreSQL + pgvector access
│   │   ├── decisioning.py    # Score → APPROVE/REVIEW/DECLINE banding
│   │   ├── features.py       # Online feature engineering (AguiarFeatureEngine)
│   │   ├── main.py           # FastAPI app, lifespan, endpoints
│   │   ├── policies.py       # Policy repository / retrieval
│   │   ├── recourse.py       # Counterfactual recourse generation
│   │   ├── schemas.py        # Pydantic request/response contracts
│   │   ├── scoring.py        # DualModelScoringEngine, KFoldTargetEncoder
│   │   └── verifier.py       # Deterministic Verifier
│   ├── data/                 # Raw + enriched Home Credit CSVs, trained model artifact
│   ├── scripts/
│   │   ├── build_kaggle_features.py   # Offline multi-table preprocessing pipeline
│   │   └── seed_policies.py           # Seeds the policy corpus for retrieval
│   ├── sql/
│   │   └── 001_init.sql      # DB schema (applicants, audit log, policy embeddings)
│   ├── tests/
│   │   ├── test_patches.py
│   │   └── test_verifier.py
│   ├── .env.example
│   ├── requirements.txt
│   └── sample_applicant.json
├── docs/
│   └── V0_FRONTEND_PROMPT.md
└── frontend/
    ├── app/                  # Next.js app router (layout, page, globals.css)
    ├── components/
    ├── hooks/
    ├── lib/
    ├── types/
    └── package.json
```

---

## Getting started

### Prerequisites

- Python 3.10+
- Node.js 18+ and `npm`
- PostgreSQL with the `pgvector` extension (optional — a local FAISS index is used by default for the prototype)
- A Google Gemini API key (optional — set `USE_MOCK_LLM=true` to run entirely without one)

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with your GEMINI_API_KEY and (optionally) DATABASE_URL

# download the Home Credit Default Risk dataset from Kaggle and place the raw
# CSVs (application_train.csv, bureau.csv, bureau_balance.csv,
# previous_application.csv, POS_CASH_balance.csv, credit_card_balance.csv,
# installments_payments.csv) into backend/data/

# one-time offline preprocessing: joins + aggregates all raw tables and
# engineers features, producing data/application_train_enriched.csv
python scripts/build_kaggle_features.py

# seed the policy corpus used by Stream B
python scripts/seed_policies.py

uvicorn app.main:app --reload
```

On first boot, if `data/model_artifact.joblib` doesn't exist yet, the scoring engine trains from scratch on the enriched dataset and persists the artifact so future restarts start in under a second.

API docs are then available at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend
npm run dev
```

`npm run dev` will install dependencies and start the dev server. Before running it, set up your own `.env` in `frontend/` (copy from `.env.example` if one is provided) with the backend API URL and any other required environment variables — this is separate from the backend's `.env`.

The console runs at `http://localhost:3000` and expects the backend at the URL configured in the frontend environment.

### Tests

```bash
cd backend
pytest
```

---

## API overview

| Endpoint | Description |
|---|---|
| `GET /health` | Model/policy store load state, active policy backend, mock vs. live LLM mode |
| `GET /api/v1/policies` | Retrieval backend and every indexed policy ID |
| `GET /api/v1/model` | Model card — version, OOF AUC (blend and per-fold), feature count, score range |
| `POST /api/v1/underwrite` | Core endpoint — scores an applicant and returns a verified audit report |
| `POST /api/v1/underwrite/verify-demo` | Negative-control demo — deliberately corrupts the LLM output to prove the verifier catches it |

**`POST /api/v1/underwrite`** takes a flat, snake_case applicant payload (income, credit request, external scores, bureau activity, free-text notes) — protected attributes are excluded by design — and returns a verified audit report: score, probability of default, status, SHAP key factors, policy citations, counterfactual recourse, verification metadata, and per-stage latency.

A failed verification returns `HTTP 422` with the specific list of failed checks rather than a degraded answer.

---

## Design principles

- **Separation of concerns.** What decides (Stream A) is never what explains (Stream B + LLM).
- **Compute → verify → commit → release.** Every stage of `/underwrite` fails independently and explicitly; nothing partially correct ever reaches the caller.
- **Audit trail is authoritative.** Logged status/limit values are recomputed deterministically from the score — never taken from the LLM's own claims.

---

## Future work

- Fairness-metric monitoring across demographic proxies not used in scoring
- Online/streaming feature updates for true real-time behavioral signals
- Versioned, timestamped policy corpus for full regulatory audit trails
- Swap the Gemini client for a native `boto3` AWS Bedrock client (interface is already shaped for it)

---

## License

Add a license for this repository (e.g. MIT) before publishing publicly.
