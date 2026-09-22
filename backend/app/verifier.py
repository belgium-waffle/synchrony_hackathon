"""The Deterministic Hallucination Verifier.

Nothing the LLM says is trusted. Every number in the payload is recomputed from
Stream A and every policy ID is checked against what Stream B actually retrieved.
A single mismatch blocks the payload — there is no partial credit and no auto-repair,
because a silently corrected audit record is worse than a rejected one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from .decisioning import credit_limit_for, decision_from_score
from .policies import RetrievedPolicy
from .schemas import LLMAuditNarrative
from .scoring import QuantResult

log = logging.getLogger(__name__)

VERIFIER_VERSION = "deterministic-verifier-1.2.0"


@dataclass
class VerificationResult:
    passed: bool
    failures: List[str] = field(default_factory=list)
    checks_run: int = 0
    narrative: Optional[LLMAuditNarrative] = None

    def render(self) -> str:
        if self.passed:
            return f"PASS — {self.checks_run} checks, every figure traceable to Stream A/B."
        return (f"FAIL ({len(self.failures)} violation(s)) — payload blocked.\n"
                + "\n".join(f"  - {f}" for f in self.failures))


class DeterministicVerifier:
    SHAP_TOL = 1e-3          # log-odds
    PCT_TOL = 0.05           # percentage points
    MONEY_TOL = 1.0          # currency units

    # Never permitted in narrative text, regardless of what the model produced.
    PROHIBITED_TERMS = ("age", "gender", "male", "female", "married", "divorced",
                        "race", "ethnicity", "nationality", "religion")

    def __init__(self, approve_threshold: int = 720, review_threshold: int = 640):
        self.approve_threshold = approve_threshold
        self.review_threshold = review_threshold

    def verify(self, raw_json: str, quant: QuantResult, policies: List[RetrievedPolicy],
               applicant: Dict[str, Any]) -> VerificationResult:
        failures: List[str] = []
        checks = 0

        # 1. Schema ----------------------------------------------------------
        checks += 1
        try:
            narrative = LLMAuditNarrative.model_validate_json(raw_json)
        except ValidationError as exc:
            return VerificationResult(False, [f"Schema validation failed: {exc}"], checks)

        # 2. Identity and score ----------------------------------------------
        checks += 2
        if narrative.applicant_id != quant.applicant_id:
            failures.append(f"applicant_id mismatch: {narrative.applicant_id!r} "
                            f"vs {quant.applicant_id!r}")
        if narrative.risk_score != quant.risk_score:
            failures.append(f"risk_score mismatch: LLM {narrative.risk_score} "
                            f"vs engine {quant.risk_score}")

        # 3. Status and limit are pure functions of the score ------------------
        checks += 2
        expected_status = decision_from_score(quant.risk_score, self.approve_threshold,
                                              self.review_threshold)
        if narrative.approval_status != expected_status:
            failures.append(f"approval_status mismatch: LLM {narrative.approval_status} "
                            f"vs policy band {expected_status}")
        expected_limit = credit_limit_for(expected_status, applicant)
        if abs(narrative.recommended_credit_limit - expected_limit) > self.MONEY_TOL:
            failures.append(f"credit limit mismatch: LLM "
                            f"{narrative.recommended_credit_limit:,.0f} vs deterministic "
                            f"{expected_limit:,.0f}")

        # 4. SHAP attributions must match to tolerance -------------------------
        truth = quant.families_dict()
        seen = set()
        for factor in narrative.key_factors:
            checks += 3
            if factor.family not in truth:
                failures.append(f"unknown feature family cited: {factor.family!r}")
                continue
            if factor.family in seen:
                failures.append(f"duplicate feature family: {factor.family!r}")
            seen.add(factor.family)

            actual = truth[factor.family]
            if abs(factor.contribution - actual.contribution) > self.SHAP_TOL:
                failures.append(f"SHAP contribution hallucinated for {factor.family}: "
                                f"LLM {factor.contribution:+.4f} vs TreeSHAP "
                                f"{actual.contribution:+.4f}")
            if abs(factor.contribution_pct - actual.contribution_pct) > self.PCT_TOL:
                failures.append(f"SHAP percentage hallucinated for {factor.family}: "
                                f"LLM {factor.contribution_pct} vs TreeSHAP "
                                f"{actual.contribution_pct}")
            if factor.direction != actual.direction:
                failures.append(f"direction inverted for {factor.family}: LLM "
                                f"{factor.direction} vs TreeSHAP {actual.direction}")

        # 5. Cited factors must be among the top-ranked families ---------------
        checks += 1
        ranked = [fa.family for fa in quant.family_attributions]
        allowed = set(ranked[: max(3, len(narrative.key_factors))])
        for factor in narrative.key_factors:
            if factor.family in truth and factor.family not in allowed:
                failures.append(f"{factor.family} cited as a key factor but ranks "
                                f"#{ranked.index(factor.family) + 1} by attribution")

        # 6. Policy IDs must come from the retrieved set -----------------------
        retrieved_ids = {p.policy_id for p in policies}
        for citation in narrative.policy_citations:
            checks += 1
            if citation.policy_id not in retrieved_ids:
                failures.append(f"policy {citation.policy_id} was never retrieved "
                                f"(retrieved: {sorted(retrieved_ids)})")

        # 7. Adverse action reasons and prohibited terms -----------------------
        checks += 2
        if narrative.approval_status in {"REVIEW", "DECLINE"} and not narrative.adverse_action_reasons:
            failures.append(f"{narrative.approval_status} requires at least one "
                            f"adverse action reason")
        if narrative.approval_status == "APPROVE" and narrative.adverse_action_reasons:
            failures.append("APPROVE must not carry adverse action reasons")

        checks += 1
        haystack = " ".join([narrative.summary, *narrative.adverse_action_reasons]).lower()
        for term in self.PROHIBITED_TERMS:
            if f" {term}" in f" {haystack}":
                failures.append(f"narrative references a protected or prohibited attribute: "
                                f"{term!r}")

        result = VerificationResult(not failures, failures, checks,
                                    narrative if not failures else None)
        if not result.passed:
            log.error("Verifier blocked payload for %s: %s", quant.applicant_id, failures)
        return result
