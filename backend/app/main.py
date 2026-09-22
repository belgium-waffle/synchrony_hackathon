"""FastAPI service entrypoint — Next-Gen Credit Intelligence.

Request flow for POST /api/v1/underwrite:

    payload -> Stream A (score + TreeSHAP)   ─┐
            -> Stream B (pgvector retrieval) ─┴─> Bedrock synthesis
            -> Deterministic Verifier -> AuditReport | 422

The verifier is the last gate before the response leaves the process. If it fails,
the caller gets a 422 carrying the violations, never the unverified narrative.
"""
from __future__ import annotations

import json
import logging
import os
import joblib
import pandas as pd
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .bedrock import MockBedrockClient, TemplateClient # Add this to your imports if not there
from .bedrock import build_prompt, build_llm_client, MockBedrockClient
from .config import Settings, get_settings
from .database import dispose_engine, init_engine, write_audit_log
from .decisioning import credit_limit_for, decision_from_score
from .features import MockApplicationData
from .policies import PolicyRepository
from .recourse import generate_recourse
from .schemas import (ApplicantPayload, AuditReport, HealthResponse, RetrievedPolicy,
                      VerificationReport)
from .scoring import MODEL_VERSION, LightGBMScoringEngine
from .verifier import VERIFIER_VERSION, DeterministicVerifier

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
log = logging.getLogger("app.main")

STATE: Dict[str, Any] = {}


# --------------------------------------------------------------- audit log ---
def _safe_parse_llm_json(raw_json: str) -> Dict[str, Any]:
    """Best-effort parse for storage. The LLM's raw output is untrusted — if it
    is not even valid JSON, log the fact rather than letting the audit write
    itself fail on a malformed JSONB cast."""
    try:
        return json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return {"raw_text": raw_json, "parse_error": True}


def _build_audit_record(*, request_id: str, quant, policies, raw_json: str,
                        recourse: Dict[str, Any], timings: Dict[str, float],
                        result, expected_status: str, expected_limit: float) -> Dict[str, Any]:
    """Assemble one row for `decision_audit_log`.

    Deliberately logs the *deterministically expected* status and limit (recomputed
    from the score, same as the verifier does) rather than whatever the LLM claimed —
    so the audit trail stays trustworthy even for a row where verification failed.
    """
    shap_payload = [
        {
            "family": fa.family,
            "contribution": fa.contribution,
            "contribution_pct": fa.contribution_pct,
            "direction": fa.direction,
            "top_features": [{"feature": c, "value": v} for c, v in fa.top_features],
        }
        for fa in quant.family_attributions
    ]
    return {
        "request_id": request_id,
        "applicant_id": quant.applicant_id,
        "model_version": MODEL_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "risk_score": quant.risk_score,
        "probability_of_default": quant.pd_probability,
        "approval_status": expected_status,
        "recommended_limit": expected_limit,
        "shap_attributions": json.dumps(shap_payload),
        "retrieved_policy_ids": [p.policy_id for p in policies],
        "llm_narrative": json.dumps(_safe_parse_llm_json(raw_json)),
        "verification_passed": result.passed,
        "verification_failures": json.dumps(result.failures),
        "recourse": json.dumps(recourse),
        "latency_ms": json.dumps(timings),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info("Starting %s (%s)", settings.APP_NAME, settings.ENVIRONMENT)

    init_engine()
    STATE["policies"] = PolicyRepository().initialise()

    t0 = time.perf_counter()
    artifact_path = "data/model_artifact.joblib"

    if os.path.exists(artifact_path):
        log.info("Found pre-trained model artifact. Loading from disk...")
        STATE["engine"] = joblib.load(artifact_path)
        log.info("Stream A ready in %.1fs (Loaded from disk)", time.perf_counter() - t0)
    else:
        log.info("No artifact found. Training on Kaggle Enriched data (this will take ~1 hour)...")
        raw = pd.read_csv("data/application_train_enriched.csv")
        STATE["engine"] = LightGBMScoringEngine(n_folds=5).fit(raw)
        
        # Save the trained model to disk so we never have to train it again
        joblib.dump(STATE["engine"], artifact_path)
        log.info("Stream A ready in %.1fs (Trained and saved to disk, OOF AUC %.4f)",
                 time.perf_counter() - t0)

    STATE["llm"] = build_llm_client()
    STATE["verifier"] = DeterministicVerifier(settings.APPROVE_THRESHOLD,
                                              settings.REVIEW_THRESHOLD)
    yield
    dispose_engine()
    STATE.clear()

app = FastAPI(
    title="Synchrony Next-Gen Credit Intelligence API",
    description=(
        "Dual-Stream credit decisioning. Stream A produces a deterministic risk score "
        "with exact TreeSHAP attribution; Stream B retrieves underwriting policy from "
        "pgvector; an LLM writes the narrative and a deterministic verifier blocks it "
        "unless every figure and citation is traceable to Stream A/B."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


# ------------------------------------------------------------------ routes ---
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    repo: PolicyRepository | None = STATE.get("policies")
    engine = STATE.get("engine")
    return HealthResponse(
        status="ok" if engine and repo else "degraded",
        model_loaded=engine is not None,
        policy_store=repo.backend if repo else "in-memory",
        policy_count=repo.policy_count if repo else 0,
        llm_mode="mock" if settings.USE_MOCK_LLM else "bedrock",
        environment=settings.ENVIRONMENT,
    )


@app.get(f"{get_settings().API_V1_PREFIX}/policies", tags=["policies"])
def list_policies():
    repo: PolicyRepository = STATE["policies"]
    return {"backend": repo.backend, "policy_ids": repo.all_policy_ids()}


@app.get(f"{get_settings().API_V1_PREFIX}/model", tags=["ops"])
def model_card():
    engine: LightGBMScoringEngine = STATE["engine"]
    return {
        "model_version": MODEL_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "oof_auc": round(engine.oof_auc_blend_, 6),
        "fold_aucs": [round(a, 6) for a in engine.fold_aucs_blend_],
        "n_features": len(engine.feature_names_),
        "score_range": [engine.SCORE_MIN, engine.SCORE_MAX],
    }


@app.post(f"{get_settings().API_V1_PREFIX}/underwrite", response_model=AuditReport,
          tags=["underwriting"],
          summary="Score an application and return a verified audit report")
def underwrite(payload: ApplicantPayload, request: Request,
               settings: Settings = Depends(get_settings)) -> AuditReport:
    engine: LightGBMScoringEngine = STATE["engine"]
    repo: PolicyRepository = STATE["policies"]
    verifier: DeterministicVerifier = STATE["verifier"]
    llm = STATE["llm"]

    applicant = payload.to_feature_row()
    timings: Dict[str, float] = {}

    # --- Stream A ---------------------------------------------------------
    t = time.perf_counter()
    try:
        quant = engine.score_applicant(applicant)
    except Exception as exc:
        log.exception("Stream A failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"Scoring engine failure: {exc.__class__.__name__}") from exc
    timings["stream_a_ms"] = round((time.perf_counter() - t) * 1000, 1)

    # --- Stream B ---------------------------------------------------------
    t = time.perf_counter()
    policies = repo.retrieve(applicant.get("UNDERWRITER_NOTES", ""),
                             k=settings.RETRIEVAL_TOP_K)
    timings["stream_b_ms"] = round((time.perf_counter() - t) * 1000, 1)
    if not policies:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "No underwriting policy could be retrieved. Underwriter notes "
                            "are required so the decision can be grounded in policy.")

    # --- Recourse (independent of the LLM) --------------------------------
    t = time.perf_counter()
    recourse = generate_recourse(engine, applicant, quant,
                                 settings.APPROVE_THRESHOLD, settings.REVIEW_THRESHOLD)
    timings["recourse_ms"] = round((time.perf_counter() - t) * 1000, 1)

    # --- LLM synthesis ----------------------------------------------------
    t = time.perf_counter()
    prompt = build_prompt(quant, policies, applicant,
                          settings.APPROVE_THRESHOLD, settings.REVIEW_THRESHOLD)
    try:
        raw_json = llm.invoke(
            prompt, quant=quant, policies=policies, applicant=applicant,
            approve_threshold=settings.APPROVE_THRESHOLD,
            review_threshold=settings.REVIEW_THRESHOLD,
        )
    except Exception as exc:
        log.exception("Bedrock invocation failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                            f"LLM synthesis unavailable: {exc.__class__.__name__}") from exc
    timings["llm_ms"] = round((time.perf_counter() - t) * 1000, 1)

    # --- Deterministic gate ------------------------------------------------
    t = time.perf_counter()
    result = verifier.verify(raw_json, quant, policies, applicant)
    timings["verifier_ms"] = round((time.perf_counter() - t) * 1000, 1)

    # --- Compliance audit trail ---------------------------------------------
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    expected_status = decision_from_score(quant.risk_score, settings.APPROVE_THRESHOLD,
                                          settings.REVIEW_THRESHOLD)
    expected_limit = credit_limit_for(expected_status, applicant)
    audit_record = _build_audit_record(
        request_id=request_id, quant=quant, policies=policies, raw_json=raw_json,
        recourse=recourse, timings=timings, result=result,
        expected_status=expected_status, expected_limit=expected_limit,
    )

    audit_write_error: Exception | None = None
    try:
        write_audit_log(audit_record)
    except RuntimeError:
        log.debug("Audit log skipped for %s: no database configured.", quant.applicant_id)
    except Exception as exc:                          # pragma: no cover
        audit_write_error = exc
        log.error("Audit log write failed for %s: %s", quant.applicant_id, exc)

    if not result.passed:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "verification_failed",
                "message": "The generated narrative could not be verified against the "
                           "model output and retrieved policy. No decision was released.",
                "failures": result.failures,
                "verifier_version": VERIFIER_VERSION,
                "risk_score": quant.risk_score,
            },
        )

    if audit_write_error is not None:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Decision computed and verified, but could not be committed to the "
            "compliance audit trail. No decision was released.",
        )

    narrative = result.narrative
    return AuditReport(
        applicant_id=quant.applicant_id,
        request_id=request_id,
        model_version=MODEL_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        risk_score=quant.risk_score,
        probability_of_default=quant.pd_probability,
        approval_status=expected_status,
        recommended_credit_limit=expected_limit,
        summary=narrative.summary,
        key_factors=narrative.key_factors,
        policy_citations=narrative.policy_citations,
        retrieved_policies=[RetrievedPolicy(policy_id=p.policy_id, title=p.title,
                                            body=p.body, similarity=p.similarity)
                            for p in policies],
        adverse_action_reasons=narrative.adverse_action_reasons,
        recourse=recourse,
        verification=VerificationReport(passed=True, checks_run=result.checks_run,
                                        failures=[], verifier_version=VERIFIER_VERSION),
        latency_ms=timings,
    )


@app.post(f"{get_settings().API_V1_PREFIX}/underwrite/verify-demo", tags=["underwriting"],
          summary="Force a hallucinated narrative to demonstrate the verifier blocking it")
def verify_demo(payload: ApplicantPayload, settings: Settings = Depends(get_settings)):
    """Negative control for the demo: identical request, tampered LLM output."""
    llm = STATE["llm"]
    if not isinstance(llm, MockBedrockClient):
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "Negative control is only available in mock LLM mode.")

    engine, repo, verifier = STATE["engine"], STATE["policies"], STATE["verifier"]
    applicant = payload.to_feature_row()
    quant = engine.score_applicant(applicant)
    policies = repo.retrieve(applicant.get("UNDERWRITER_NOTES", ""), k=settings.RETRIEVAL_TOP_K)
    raw_json = llm.invoke("", quant=quant, policies=policies, applicant=applicant,
                          approve_threshold=settings.APPROVE_THRESHOLD,
                          review_threshold=settings.REVIEW_THRESHOLD, hallucinate=True)
    result = verifier.verify(raw_json, quant, policies, applicant)
    return JSONResponse(status_code=200, content={
        "verification_passed": result.passed,
        "checks_run": result.checks_run,
        "failures": result.failures,
        "payload_released": result.narrative is not None,
        "verifier_version": VERIFIER_VERSION,
    })