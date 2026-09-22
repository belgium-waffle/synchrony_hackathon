"""Deliverable 2 — Counterfactual Recourse.

Takes the applicant's most risk-increasing TreeSHAP feature, perturbs it to a
healthy baseline, re-scores through the *same* model, and reports the realised
score delta.

Two design rules make this Responsible AI rather than a demo trick:

1. **Only actionable features are candidates.** A recourse the applicant cannot act
   on is not recourse. `ACTIONABLE_FEATURES` is an allow-list; anything absent from
   it — age, gender, family status, social-circle defaults, region rating — can never
   be surfaced, so the service cannot tell someone their score would improve if they
   were older or a different gender. Under ECOA that output would itself be evidence
   of a disparate-treatment problem.

2. **The delta is measured, never estimated.** We do not read the score change off the
   SHAP value. We mutate the feature, run a full forward pass through the fold-bagged
   ensemble, and report the actual difference. SHAP is a local linear attribution; the
   true response to a large perturbation is non-linear and the two disagree, sometimes
   materially.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .decisioning import decision_from_score

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecourseRule:
    """How to improve one feature, and how to say so in plain language."""
    display_name: str
    target_value: float
    horizon_months: int
    phrasing: str                 # "{action}" clause, e.g. "reduce your credit enquiries to 0"
    better_is_lower: bool = True


# Allow-list. A feature absent from this map is structurally ineligible for recourse.
ACTIONABLE_FEATURES: Dict[str, RecourseRule] = {
    "AMT_REQ_CREDIT_BUREAU_YEAR": RecourseRule(
        display_name="Credit bureau enquiries (12m)", target_value=0.0, horizon_months=12,
        phrasing="hold new credit enquiries at zero for the next 12 months",
    ),
    "ANNUITY_INCOME_PERC": RecourseRule(
        display_name="Annuity-to-income ratio", target_value=0.25, horizon_months=6,
        phrasing="bring your annuity-to-income ratio down to 0.25, by requesting a longer term "
                 "or a smaller principal",
    ),
    "PAYMENT_RATE": RecourseRule(
        display_name="Payment rate", target_value=0.035, horizon_months=6,
        phrasing="extend the repayment term so the payment rate falls to roughly 0.035",
    ),
    "INCOME_CREDIT_PERC": RecourseRule(
        display_name="Income-to-credit ratio", target_value=0.45, horizon_months=6,
        phrasing="reduce the requested credit amount so it is closer to twice your annual income",
        better_is_lower=False,
    ),
    "AMT_CREDIT": RecourseRule(
        display_name="Requested credit amount", target_value=float("nan"), horizon_months=0,
        phrasing="reduce the requested credit amount by 20%",
    ),
    "DAYS_EMPLOYED": RecourseRule(
        display_name="Employment tenure", target_value=-365.0 * 2, horizon_months=12,
        phrasing="reach two years of continuous verifiable employment",
        better_is_lower=False,
    ),
    "EXT_SOURCE_1": RecourseRule(
        display_name="External score 1 (currently missing)", target_value=0.55, horizon_months=6,
        phrasing="complete the file by consenting to the additional external score check",
        better_is_lower=False,
    ),
    "EXT_SOURCE_3": RecourseRule(
        display_name="External score 3 (currently missing)", target_value=0.55, horizon_months=6,
        phrasing="complete the file by consenting to the additional external score check",
        better_is_lower=False,
    ),
}

# Explicit deny-list, documented so an auditor can see the exclusions were deliberate
# rather than accidental omissions from the allow-list above.
PROTECTED_OR_IMMUTABLE = {
    "CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE", "REGION_RATING_CLIENT", "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
}

# Ratio features are derived, not stored. To perturb them we must move the inputs
# they are computed from, otherwise AguiarFeatureEngine recomputes and overwrites us.
DERIVED_FEATURE_DRIVERS: Dict[str, str] = {
    "ANNUITY_INCOME_PERC": "AMT_ANNUITY",
    "PAYMENT_RATE": "AMT_ANNUITY",
    "INCOME_CREDIT_PERC": "AMT_CREDIT",
    "DAYS_EMPLOYED_PERC": "DAYS_EMPLOYED",
}


def _base_column(engineered_column: str) -> str:
    """Strip get_dummies suffixes back to the source column name."""
    for base in sorted(set(ACTIONABLE_FEATURES) | PROTECTED_OR_IMMUTABLE, key=len, reverse=True):
        if engineered_column == base or engineered_column.startswith(base + "_"):
            return base
    return engineered_column


def _apply_counterfactual(applicant: Dict[str, Any], feature: str,
                          rule: RecourseRule) -> Optional[Dict[str, Any]]:
    """Return a copy of the applicant with the counterfactual applied, or None."""
    cf = dict(applicant)
    income = float(applicant.get("AMT_INCOME_TOTAL") or 0.0)
    credit = float(applicant.get("AMT_CREDIT") or 0.0)

    if feature == "ANNUITY_INCOME_PERC":
        if income <= 0:
            return None
        cf["AMT_ANNUITY"] = round(income * rule.target_value, 2)
    elif feature == "PAYMENT_RATE":
        if credit <= 0:
            return None
        cf["AMT_ANNUITY"] = round(credit * rule.target_value, 2)
    elif feature == "INCOME_CREDIT_PERC":
        if income <= 0:
            return None
        cf["AMT_CREDIT"] = round(income / rule.target_value, 2)
    elif feature == "AMT_CREDIT":
        cf["AMT_CREDIT"] = round(credit * 0.80, 2)
    else:
        cf[feature] = rule.target_value
    return cf


def generate_recourse(engine, applicant: Dict[str, Any], quant,
                      approve_threshold: int, review_threshold: int) -> Dict[str, Any]:
    """Build the counterfactual recourse block for the audit report.

    `engine` is a fitted LightGBMScoringEngine; `quant` its QuantResult for this applicant.
    """
    # Rank engineered columns by how much they pushed risk UP, then keep only those
    # whose source column is on the actionable allow-list.
    candidates = []
    for column, shap_value in sorted(quant.raw_shap.items(), key=lambda kv: kv[1], reverse=True):
        if shap_value <= 0:
            break                                    # remaining features reduce risk
        base = _base_column(column)
        if base in PROTECTED_OR_IMMUTABLE:
            log.debug("Skipping %s for recourse: protected or immutable.", base)
            continue
        if base in ACTIONABLE_FEATURES:
            candidates.append((base, shap_value))

    if not candidates:
        return {
            "available": False,
            "advice": ("No actionable recourse is available for this application. The factors "
                       "driving the outcome are either already at a healthy level or are not "
                       "attributes the applicant can act on."),
        }

    feature, _ = candidates[0]
    rule = ACTIONABLE_FEATURES[feature]
    cf_applicant = _apply_counterfactual(applicant, feature, rule)
    if cf_applicant is None:
        return {"available": False,
                "advice": "Recourse could not be computed for this application."}

    # Measure the delta: full forward pass, not a SHAP extrapolation.
    cf_prob = engine.predict_proba_row(cf_applicant)
    cf_score = engine.probability_to_score(cf_prob)
    delta = cf_score - quant.risk_score

    current_status = decision_from_score(quant.risk_score, approve_threshold, review_threshold)
    projected_status = decision_from_score(cf_score, approve_threshold, review_threshold)
    crossed = projected_status if projected_status != current_status else None

    current_value = applicant.get(feature)
    try:
        current_value = None if current_value is None or (
            isinstance(current_value, float) and np.isnan(current_value)) else float(current_value)
    except (TypeError, ValueError):
        current_value = None

    if delta <= 0:
        advice = (f"Actionable Advice: adjusting {rule.display_name.lower()} alone would not "
                  f"improve this score (projected change {delta:+d} points). The outcome is "
                  f"driven by factors outside this single input.")
    else:
        crossing = (f", crossing the {crossed} threshold" if crossed else
                    f", which would not by itself change the {current_status} outcome")
        horizon = (f" over the next {rule.horizon_months} months"
                   if rule.horizon_months else "")
        advice = (f"Actionable Advice: if you {rule.phrasing}{horizon}, your score is projected "
                  f"to rise by {delta} points to {cf_score}{crossing}.")

    return {
        "available": True,
        "feature": feature,
        "display_name": rule.display_name,
        "current_value": current_value,
        "target_value": None if np.isnan(rule.target_value) else float(rule.target_value),
        "projected_score": cf_score,
        "score_delta": delta,
        "crosses_threshold": crossed,
        "horizon_months": rule.horizon_months or None,
        "advice": advice,
    }
