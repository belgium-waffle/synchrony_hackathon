-- Synchrony Next-Gen Credit Intelligence — PostgreSQL + pgvector schema
-- Run once: psql "$DATABASE_URL" -f sql/001_init.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- Stream B --
-- Underwriting policy corpus. `embedding` dimensionality MUST equal
-- settings.EMBEDDING_DIM (384 for all-MiniLM-L6-v2). Changing the embedding
-- model is a schema migration, not a config tweak.
CREATE TABLE IF NOT EXISTS underwriting_policies (
    policy_id     TEXT PRIMARY KEY,
    title         TEXT        NOT NULL,
    body          TEXT        NOT NULL,
    version       TEXT        NOT NULL DEFAULT 'v1',
    is_active     BOOLEAN     NOT NULL DEFAULT TRUE,
    embedding     vector(384) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT policy_id_format CHECK (policy_id ~ '^UW-[0-9]+\.[0-9]+$')
);

-- HNSW beats IVFFlat for a corpus this size and needs no training step.
-- vector_cosine_ops matches the <=> operator used in PolicyRepository.
CREATE INDEX IF NOT EXISTS idx_policies_embedding_hnsw
    ON underwriting_policies USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_policies_active
    ON underwriting_policies (is_active) WHERE is_active = TRUE;

-- ------------------------------------------------------------- audit trail --
-- Every released decision is persisted. This table is the regulatory artifact:
-- it stores the exact SHAP vector and the retrieved policy IDs alongside the
-- narrative, so any decision can be reconstructed and re-verified years later.
CREATE TABLE IF NOT EXISTS decision_audit_log (
    id                    BIGSERIAL PRIMARY KEY,
    request_id            UUID        NOT NULL,
    applicant_id          TEXT        NOT NULL,
    model_version         TEXT        NOT NULL,
    verifier_version      TEXT        NOT NULL,
    risk_score            INTEGER     NOT NULL CHECK (risk_score BETWEEN 300 AND 850),
    probability_of_default NUMERIC(8,6) NOT NULL,
    approval_status       TEXT        NOT NULL
                          CHECK (approval_status IN ('APPROVE','REVIEW','DECLINE')),
    recommended_limit     NUMERIC(14,2) NOT NULL,
    shap_attributions     JSONB       NOT NULL,
    retrieved_policy_ids  TEXT[]      NOT NULL,
    llm_narrative         JSONB       NOT NULL,
    verification_passed   BOOLEAN     NOT NULL,
    verification_failures  JSONB      NOT NULL DEFAULT '[]'::jsonb,
    recourse              JSONB,
    latency_ms            JSONB,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_applicant ON decision_audit_log (applicant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_blocked   ON decision_audit_log (created_at DESC)
    WHERE verification_passed = FALSE;

-- Blocked payloads are retained deliberately: the rate at which the verifier
-- fires is a model-governance metric, not noise to be discarded.
COMMENT ON TABLE decision_audit_log IS
    'Immutable decision record. Append-only; never UPDATE a released decision.';
