"""The OpenAPI contract.

These models are the single source of truth for the API surface. FastAPI renders
them into /docs, and the React client is generated against that schema.
"""
from __future__ import annotations

import logging
import re
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

log = logging.getLogger(__name__)

DecisionStatus = Literal["APPROVE", "REVIEW", "DECLINE"]
RiskDirection = Literal["increases_risk", "decreases_risk"]

# Fixed baseline values for attributes the intake UI deliberately does not collect.
#
# Stream A's fitted column space (AguiarFeatureEngine) expects these columns to be
# present, so they cannot simply be omitted — Pandas would KeyError and the request
# would 500. But a default value alone only protects a well-behaved client; nothing
# stops a caller from hitting the API directly with `code_gender: "M"`.
#
# So this is enforced two ways:
#   1. A default so an omitted field never causes a 422/500.
#   2. A model_validator (below) that OVERWRITES these fields to the fixed baseline
#      regardless of what was submitted, so the exclusion is a backend guarantee,
#      not a frontend convention a caller could bypass.
#
# Because every applicant is scored with the identical value for these columns, they
# carry zero variance across production traffic. A feature with zero variance can
# never register a nonzero TreeSHAP attribution for any individual decision — the
# constraint isn't just "hidden from the form", it is "structurally incapable of
# influencing an individual outcome." Values below are each the training-data mode,
# chosen so pinning them doesn't push any applicant into an out-of-distribution corner
# of the one-hot space.
PROTECTED_ATTRIBUTE_DEFAULTS: dict[str, Any] = {
    "code_gender": "F",
    "flag_own_car": "N",
    "flag_own_realty": "Y",
    "name_family_status": "Married",
    "name_housing_type": "House / apartment",
}


# ---------------------------------------------------------------- request ----
class ApplicantPayload(BaseModel):
    """Incoming application. Field names mirror the Home Credit schema so the same
    payload shape works against the real `application_train` data."""

    model_config = {"json_schema_extra": {"example": {
        "sk_id_curr": 100012,
        "amt_income_total": 180000.0,
        "amt_credit": 675000.0,
        "amt_annuity": 32000.0,
        "amt_goods_price": 607500.0,
        "cnt_fam_members": 2,
        "days_birth": -12005,
        "days_employed": 365243,
        "days_registration": -4200.0,
        "days_id_publish": -2100,
        "days_last_phone_change": -310.0,
        "ext_source_1": None,
        "ext_source_2": 0.31,
        "ext_source_3": None,
        "code_gender": "F",
        "flag_own_car": "N",
        "flag_own_realty": "Y",
        "name_contract_type": "Cash loans",
        "name_income_type": "Working",
        "name_education_type": "Secondary / secondary special",
        "name_family_status": "Married",
        "name_housing_type": "House / apartment",
        "occupation_type": "Laborers",
        "region_rating_client": 2,
        "obs_30_cnt_social_circle": 3.0,
        "def_30_cnt_social_circle": 2.0,
        "amt_req_credit_bureau_year": 6.0,
        "underwriter_notes": "Applicant is not currently employed and two of three external scoring sources are missing from the file.",
    }}}

    sk_id_curr: int = Field(description="Applicant identifier")
    amt_income_total: float = Field(gt=0, description="Total annual income")
    amt_credit: float = Field(gt=0, description="Credit amount of the loan")
    amt_annuity: Optional[float] = Field(default=None, ge=0, description="Loan annuity")
    amt_goods_price: Optional[float] = Field(default=None, ge=0)
    cnt_fam_members: float = Field(default=1, ge=1)

    days_birth: int = Field(lt=0, description="Age in days, negative relative to application")
    days_employed: float = Field(
        description="Employment length in days. 365243 is the Home Credit 'not employed' sentinel."
    )
    days_registration: float = Field(default=-1000.0, le=0)
    days_id_publish: float = Field(default=-1000.0, le=0)
    days_last_phone_change: float = Field(default=-500.0, le=0)

    ext_source_1: Optional[float] = Field(default=None, ge=0, le=1)
    ext_source_2: Optional[float] = Field(default=None, ge=0, le=1)
    ext_source_3: Optional[float] = Field(default=None, ge=0, le=1)

    # --- protected / immutable attributes -----------------------------------
    # Excluded from the intake UI by design (see PROTECTED_ATTRIBUTE_DEFAULTS above).
    # Any value submitted here is discarded and replaced with the fixed baseline —
    # see `_pin_protected_attributes` below. They are declared as regular fields
    # (not `Literal[default]`) so a well-behaved client omitting them entirely still
    # produces a valid request; the enforcement happens after parsing, not via the
    # type system.
    code_gender: str = Field(default=PROTECTED_ATTRIBUTE_DEFAULTS["code_gender"])
    flag_own_car: str = Field(default=PROTECTED_ATTRIBUTE_DEFAULTS["flag_own_car"])
    flag_own_realty: str = Field(default=PROTECTED_ATTRIBUTE_DEFAULTS["flag_own_realty"])
    name_family_status: str = Field(default=PROTECTED_ATTRIBUTE_DEFAULTS["name_family_status"])
    name_housing_type: str = Field(default=PROTECTED_ATTRIBUTE_DEFAULTS["name_housing_type"])

    # --- collected attributes ------------------------------------------------
    name_contract_type: str = "Cash loans"
    name_income_type: str = "Working"
    name_education_type: str = "Secondary / secondary special"
    occupation_type: Optional[str] = "Laborers"

    region_rating_client: int = Field(default=2, ge=1, le=3)
    obs_30_cnt_social_circle: float = Field(default=0.0, ge=0)
    def_30_cnt_social_circle: float = Field(default=0.0, ge=0)
    amt_req_credit_bureau_year: float = Field(default=0.0, ge=0)

    underwriter_notes: str = Field(
        default="", max_length=4000,
        description="Free text. Feeds Stream B retrieval only — never the scoring model.",
    )

    @model_validator(mode="after")
    def _pin_protected_attributes(self) -> "ApplicantPayload":
        """Overwrite any client-supplied value with the fixed baseline.

        This runs on every request, including ones built entirely from the intake
        UI (where it is a no-op, since the UI never sends these fields and the
        defaults already match). It only does real work against a caller that
        bypasses the UI — which is exactly the case it exists to cover.
        """
        for field_name, baseline in PROTECTED_ATTRIBUTE_DEFAULTS.items():
            submitted = getattr(self, field_name)
            if submitted != baseline:
                log.warning(
                    "Protected attribute override rejected: %s=%r submitted for "
                    "applicant %s; pinned to baseline %r.",
                    field_name, submitted, self.sk_id_curr, baseline,
                )
                setattr(self, field_name, baseline)
        return self

    def to_feature_row(self) -> dict:
        """Uppercase back to the Home Credit column names the model was trained on."""
        row = {k.upper(): v for k, v in self.model_dump().items()}
        row["UNDERWRITER_NOTES"] = row.pop("UNDERWRITER_NOTES", "")
        return row


# --------------------------------------------------------------- response ----
class FeatureFamilyAttribution(BaseModel):
    family: str
    contribution: float = Field(description="Summed TreeSHAP value, log-odds space")
    contribution_pct: float = Field(ge=0, le=100)
    direction: RiskDirection
    top_features: List[str] = Field(default_factory=list)


class PolicyCitation(BaseModel):
    policy_id: str
    why_relevant: str = Field(min_length=10, max_length=400)

    @field_validator("policy_id")
    @classmethod
    def _format(cls, v: str) -> str:
        if not re.fullmatch(r"UW-\d+\.\d+", v):
            raise ValueError(f"Malformed policy id: {v!r}")
        return v


class RetrievedPolicy(BaseModel):
    policy_id: str
    title: str
    body: str
    similarity: float


class CounterfactualRecourse(BaseModel):
    """Deliverable 2 — actionable, non-discriminatory recourse."""
    available: bool
    feature: Optional[str] = None
    display_name: Optional[str] = None
    current_value: Optional[float] = None
    target_value: Optional[float] = None
    projected_score: Optional[int] = None
    score_delta: Optional[int] = None
    crosses_threshold: Optional[DecisionStatus] = None
    horizon_months: Optional[int] = None
    advice: str


class LLMAuditNarrative(BaseModel):
    """Exactly what the LLM is permitted to emit. Verified before release."""
    applicant_id: str
    risk_score: int = Field(ge=300, le=850)
    approval_status: DecisionStatus
    recommended_credit_limit: float = Field(ge=0)
    summary: str = Field(min_length=20, max_length=1200)
    key_factors: List[FeatureFamilyAttribution] = Field(min_length=1, max_length=5)
    policy_citations: List[PolicyCitation] = Field(min_length=1, max_length=4)
    adverse_action_reasons: List[str] = Field(default_factory=list, max_length=4)


class VerificationReport(BaseModel):
    passed: bool
    checks_run: int
    failures: List[str] = Field(default_factory=list)
    verifier_version: str


class AuditReport(BaseModel):
    """The only object this service ever returns to a caller."""
    applicant_id: str
    request_id: str
    model_version: str
    generated_at: str

    risk_score: int = Field(ge=300, le=850)
    probability_of_default: float = Field(ge=0, le=1)
    approval_status: DecisionStatus
    recommended_credit_limit: float = Field(ge=0)

    summary: str
    key_factors: List[FeatureFamilyAttribution]
    policy_citations: List[PolicyCitation]
    retrieved_policies: List[RetrievedPolicy]
    adverse_action_reasons: List[str]
    recourse: CounterfactualRecourse
    verification: VerificationReport

    latency_ms: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    policy_store: Literal["pgvector", "in-memory"]
    policy_count: int
    llm_mode: Literal["bedrock", "mock"]
    environment: str
