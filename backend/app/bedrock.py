"""LLM synthesis layer — Native Gemini API.

The LLM's only job is prose. Score, status and limit are computed upstream and passed
in; the model is instructed to reproduce them, and the verifier independently
recomputes them afterwards. A compromised or hallucinating model cannot move a number.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Protocol

from .config import get_settings
from .decisioning import credit_limit_for, decision_from_score
from .policies import RetrievedPolicy
from .scoring import QuantResult

log = logging.getLogger(__name__)

def build_prompt(quant: QuantResult, policies: List[RetrievedPolicy],
                 applicant: Dict[str, Any], approve_threshold: int,
                 review_threshold: int) -> str:
    drivers = "\n".join(
        f"  - {fa.family}: contribution={fa.contribution:+.4f} "
        f"({fa.contribution_pct}% of total impact, {fa.direction}); "
        f"top features: {', '.join(f'{c} {v:+.4f}' for c, v in fa.top_features)}"
        for fa in quant.family_attributions
    )
    policy_block = "\n".join(f"  [{p.policy_id}] {p.title}\n    {p.body}" for p in policies)
    status = decision_from_score(quant.risk_score, approve_threshold, review_threshold)
    limit = credit_limit_for(status, applicant)

    return f"""You are a credit underwriting analyst. Write an audit-ready explanation.

You are NOT deciding this application. The score, status and limit below were computed
deterministically upstream and must be reproduced verbatim.

APPLICANT: {quant.applicant_id}
RISK SCORE (300-850): {quant.risk_score}
MODEL PD: {quant.pd_probability:.4f}
APPROVAL STATUS (fixed): {status}
RECOMMENDED CREDIT LIMIT (fixed): {limit}

SHAP FEATURE-FAMILY ATTRIBUTIONS (log-odds; positive raises default risk):
{drivers}

UNDERWRITER NOTES:
  {applicant.get('UNDERWRITER_NOTES', '')}

RETRIEVED UNDERWRITING POLICY:
{policy_block}

RULES
1. Copy risk_score, approval_status and recommended_credit_limit exactly as given.
2. In key_factors, use only the family names above, with their exact contribution,
   contribution_pct and direction values. Do not round, rescale or invent numbers.
3. In policy_citations, cite only from: {[p.policy_id for p in policies]}.
4. adverse_action_reasons must be non-empty for REVIEW or DECLINE and must reference
   only families whose direction is increases_risk.
5. Never reference age, gender, marital status or any proxy for them.
6. Respond with a single JSON object with keys: applicant_id, risk_score,
   approval_status, recommended_credit_limit, summary, key_factors,
   policy_citations, adverse_action_reasons. No prose, no markdown fences.
"""

class LLMClient(Protocol):
    def invoke(self, prompt: str, **context: Any) -> str: ...

class GeminiClient:
    """Real Google Gemini API invocation using the native SDK."""

    def __init__(self):
        import google.generativeai as genai
        from dotenv import load_dotenv
        
        load_dotenv()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please check your backend/.env file.")

        self.settings = get_settings()
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel("gemini-3.5-flash")

    def invoke(self, prompt: str, **_: Any) -> str:
        response = self._model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": self.settings.BEDROCK_TEMPERATURE
            }
        )
        text_out = response.text.strip()
        
        if text_out.startswith("```"):
            text_out = text_out.strip("```json").strip("```").strip()
            if "{" in text_out and "}" in text_out:
                text_out = text_out[text_out.find("{"):text_out.rfind("}") + 1]
        return text_out

class MockBedrockClient:
    """Deterministic stand-in with the same call signature."""

    def invoke(self, prompt: str, *, quant: QuantResult, policies: List[RetrievedPolicy],
               applicant: Dict[str, Any], approve_threshold: int = 720,
               review_threshold: int = 640, hallucinate: bool = False, **_: Any) -> str:
        status = decision_from_score(quant.risk_score, approve_threshold, review_threshold)
        limit = credit_limit_for(status, applicant)
        top3 = quant.family_attributions[:3]

        factors = [{"family": fa.family, "contribution": fa.contribution,
                    "contribution_pct": fa.contribution_pct, "direction": fa.direction,
                    "top_features": [c for c, _ in fa.top_features]} for fa in top3]
        citations = [{"policy_id": p.policy_id,
                      "why_relevant": f"{p.title} applies to the evidence in the underwriter notes."}
                      for p in policies]
        adverse = [f"{fa.family} weighed against this application."
                   for fa in top3 if fa.direction == "increases_risk"]
        if status != "APPROVE" and not adverse:
            adverse = ["Overall model score fell below the auto-approval threshold."]

        if hallucinate:
            factors[0]["contribution"] = round(factors[0]["contribution"] + 0.37, 4)
            citations.append({"policy_id": "UW-9.9",
                              "why_relevant": "Fabricated clause that was never retrieved."})
            limit = limit * 1.5 + 5000

        return json.dumps({
            "applicant_id": quant.applicant_id,
            "risk_score": quant.risk_score,
            "approval_status": status,
            "recommended_credit_limit": limit,
            "summary": (
                f"Applicant {quant.applicant_id} scores {quant.risk_score} on a 300-850 scale "
                f"(model PD {quant.pd_probability:.2%}). The dominant driver is {top3[0].family}, "
                f"which {top3[0].direction.replace('_', ' ')} and accounts for "
                f"{top3[0].contribution_pct}% of total attribution. Status: {status}."
            ),
            "key_factors": factors,
            "policy_citations": citations,
            "adverse_action_reasons": adverse if status != "APPROVE" else [],
        })

class TemplateClient:
    """Deterministic narrative generation using rule-based templating."""

    def invoke(self, prompt: str, *, quant: QuantResult, policies: List[RetrievedPolicy],
               applicant: Dict[str, Any], approve_threshold: int = 720,
               review_threshold: int = 640, **_: Any) -> str:
        
        status = decision_from_score(quant.risk_score, approve_threshold, review_threshold)
        limit = credit_limit_for(status, applicant)
        top3 = quant.family_attributions[:3]

        # 1. Template the key factors
        factors = [{"family": fa.family, "contribution": fa.contribution,
                    "contribution_pct": fa.contribution_pct, "direction": fa.direction,
                    "top_features": [c for c, _ in fa.top_features]} for fa in top3]
        
        # 2. Template the policy citations
        citations = [{"policy_id": p.policy_id,
                      "why_relevant": f"Automated retrieval matched {p.policy_id}: {p.title} to applicant evidence."}
                     for p in policies]
        
        # 3. Template the adverse action reasons (ECOA compliance)
        adverse = [f"The applicant's {fa.family} fell outside acceptable parameters, increasing risk."
                   for fa in top3 if fa.direction == "increases_risk"]
        if status != "APPROVE" and not adverse:
            adverse = ["Overall model score fell below the auto-approval threshold."]

        # 4. Template the summary narrative
        summary_template = (
            f"Applicant {quant.applicant_id} received a risk score of {quant.risk_score} "
            f"(Model PD: {quant.pd_probability:.2%}). The primary factor driving this "
            f"decision was {top3[0].family}, accounting for {top3[0].contribution_pct}% "
            f"of the total risk attribution. The application has been marked as {status}."
        )

        # Return the exact JSON schema the verifier expects
        return json.dumps({
            "applicant_id": quant.applicant_id,
            "risk_score": quant.risk_score,
            "approval_status": status,
            "recommended_credit_limit": limit,
            "summary": summary_template,
            "key_factors": factors,
            "policy_citations": citations,
            "adverse_action_reasons": adverse if status != "APPROVE" else [],
        })

def build_llm_client() -> LLMClient:
    settings = get_settings()
    if settings.USE_MOCK_LLM:
        log.info("LLM mode: template / mock (rule-based deterministic narrative)")
        return TemplateClient()
    log.info("LLM mode: Google Gemini (Native SDK)")
    return GeminiClient()