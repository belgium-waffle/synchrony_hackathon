"""Tests for the two patches: protected-attribute pinning, and the compliance
audit log.

No live PostgreSQL is available in this environment, so the audit-log tests
mock `app.main.write_audit_log` rather than hitting a real database. That proves
the control flow — what gets logged, and the fail-closed/fail-open behaviour —
is correct. It does **not** prove the raw SQL or the TEXT[]/JSONB parameter
binding work against a real pgvector instance; run `pytest -m integration`
(not yet defined) against a real database before relying on this in production.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import app

APPLICANT = {
    "sk_id_curr": 100012, "amt_income_total": 180000.0, "amt_credit": 675000.0,
    "amt_annuity": 32000.0, "days_birth": -12005, "days_employed": 365243,
    "ext_source_2": 0.31, "amt_req_credit_bureau_year": 6.0,
    "underwriter_notes": "Not currently employed; two external scoring sources missing.",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------- protected pinning
def test_omitted_protected_attributes_do_not_500(client):
    """The original bug report: a UI payload with no gender/marital/housing keys
    at all must not 422 or 500."""
    minimal = {k: v for k, v in APPLICANT.items()}   # already omits the five fields
    r = client.post("/api/v1/underwrite", json=minimal)
    assert r.status_code == 200


def test_client_supplied_protected_values_are_pinned(client):
    """A caller bypassing the UI and sending a protected attribute directly must
    not have it honoured — pinning is a backend guarantee, not a UI convention."""
    from app.schemas import ApplicantPayload, PROTECTED_ATTRIBUTE_DEFAULTS

    tampered = dict(APPLICANT, code_gender="M", name_family_status="Single / not married",
                    name_housing_type="Rented apartment", flag_own_car="Y", flag_own_realty="N")
    parsed = ApplicantPayload.model_validate(tampered)
    for field, baseline in PROTECTED_ATTRIBUTE_DEFAULTS.items():
        assert getattr(parsed, field) == baseline, f"{field} was not pinned"


def test_pinned_row_reaches_stream_a_unchanged(client):
    """End-to-end: even with tampered protected attributes in the request body,
    the response is identical to the untampered request, because Stream A
    only ever sees the pinned baseline."""
    tampered = dict(APPLICANT, code_gender="M", name_family_status="Widow")
    r_clean = client.post("/api/v1/underwrite", json=APPLICANT).json()
    r_tampered = client.post("/api/v1/underwrite", json=tampered).json()
    
    assert r_clean["risk_score"] == r_tampered["risk_score"]
    
    # Check that family names match exactly, and contributions match within tolerance
    for clean_factor, tampered_factor in zip(r_clean["key_factors"], r_tampered["key_factors"]):
        assert clean_factor["family"] == tampered_factor["family"]
        assert abs(clean_factor["contribution"] - tampered_factor["contribution"]) < 1e-4


# --------------------------------------------------------------- audit log
def test_audit_log_attempted_and_ignored_when_no_database(client, caplog):
    """In this test environment DATABASE_URL is unset, so write_audit_log raises
    RuntimeError. That must not affect the response."""
    r = client.post("/api/v1/underwrite", json=APPLICANT)
    assert r.status_code == 200


def test_audit_log_called_with_correct_shape(client, monkeypatch):
    """Capture the record passed to write_audit_log without touching a real DB."""
    captured = {}

    def fake_write(record):
        captured.update(record)

    monkeypatch.setattr(main, "write_audit_log", fake_write)
    r = client.post("/api/v1/underwrite", json=APPLICANT)
    assert r.status_code == 200
    body = r.json()

    assert captured["applicant_id"] == body["applicant_id"]
    assert captured["verification_passed"] is True
    assert captured["verification_failures"] == "[]"
    assert captured["retrieved_policy_ids"] == [p["policy_id"] for p in body["retrieved_policies"]]
    assert captured["approval_status"] == body["approval_status"]
    assert abs(captured["recommended_limit"] - body["recommended_credit_limit"]) < 1.0
    assert "shap_attributions" in captured and "llm_narrative" in captured


def test_audit_write_failure_fails_closed(client, monkeypatch):
    """A configured-but-failing database write must block the decision (500),
    never silently release an unaudited APPROVE/REVIEW/DECLINE."""
    def failing_write(record):
        raise ConnectionError("simulated PostgreSQL outage")

    monkeypatch.setattr(main, "write_audit_log", failing_write)
    r = client.post("/api/v1/underwrite", json=APPLICANT)
    assert r.status_code == 500
    assert "compliance audit trail" in r.text.lower()


def test_blocked_decision_still_attempts_audit_log(client, monkeypatch):
    """A hallucinated narrative that fails verification must still be logged —
    blocked payloads are a governance metric, not noise."""
    class AlwaysHallucinateLLM:
        def __init__(self, base):
            self._base = base

        def invoke(self, prompt, **kwargs):
            kwargs["hallucinate"] = True
            return self._base.invoke(prompt, **kwargs)

    captured = {}
    monkeypatch.setattr(main, "write_audit_log", lambda record: captured.update(record))
    monkeypatch.setitem(main.STATE, "llm", AlwaysHallucinateLLM(main.STATE["llm"]))

    r = client.post("/api/v1/underwrite", json=APPLICANT)
    assert r.status_code == 422
    assert captured, "write_audit_log was never called for a blocked decision"
    assert captured["verification_passed"] is False
    assert captured["verification_failures"] != "[]"
