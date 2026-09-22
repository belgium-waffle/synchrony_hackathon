"""Deterministic decisioning.

Status and credit line are pure functions of the score and the applicant record.
The LLM is handed the results and told to reproduce them; the verifier recomputes
them independently. Nothing here is ever delegated to a model.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

LIMIT_MULTIPLIER = {"APPROVE": 2.0, "REVIEW": 0.75, "DECLINE": 0.0}
UW_5_1_DTI_CAP = 0.45


def decision_from_score(score: int, approve_threshold: int = 720,
                        review_threshold: int = 640) -> str:
    if score >= approve_threshold:
        return "APPROVE"
    if score >= review_threshold:
        return "REVIEW"
    return "DECLINE"


def credit_limit_for(status: str, applicant: Dict[str, Any]) -> float:
    """Deterministic line assignment, including the UW-5.1 leverage cap."""
    income = float(applicant["AMT_INCOME_TOTAL"])
    monthly_income = income / 12.0
    limit = monthly_income * LIMIT_MULTIPLIER[status]
    annuity = applicant.get("AMT_ANNUITY")
    if annuity is not None and not pd.isna(annuity):
        if float(annuity) / income > UW_5_1_DTI_CAP:
            limit = min(limit, monthly_income)       # UW-5.1
    return float(round(limit, -2))
