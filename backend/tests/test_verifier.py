"""The verifier is the safety-critical component, so it gets the tests.

Each case proves one hallucination class is caught. If any of these ever go green
when they should be red, the service is shipping unverified decisions.
"""
import json

import pytest
from fastapi.testclient import TestClient

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


def test_clean_request_passes_verification(client):
    r = client.post("/api/v1/underwrite", json=APPLICANT)
    assert r.status_code == 200
    body = r.json()
    assert body["verification"]["passed"] is True
    assert body["verification"]["failures"] == []
    assert 300 <= body["risk_score"] <= 850


def test_narrative_numbers_match_treeshap(client):
    body = client.post("/api/v1/underwrite", json=APPLICANT).json()
    total = sum(f["contribution_pct"] for f in body["key_factors"])
    assert 0 < total <= 100.01
    for factor in body["key_factors"]:
        expected = "increases_risk" if factor["contribution"] > 0 else "decreases_risk"
        assert factor["direction"] == expected


def test_citations_subset_of_retrieved(client):
    body = client.post("/api/v1/underwrite", json=APPLICANT).json()
    retrieved = {p["policy_id"] for p in body["retrieved_policies"]}
    cited = {c["policy_id"] for c in body["policy_citations"]}
    assert cited.issubset(retrieved)


def test_hallucinated_payload_is_blocked(client):
    r = client.post("/api/v1/underwrite/verify-demo", json=APPLICANT)
    assert r.status_code == 200
    body = r.json()
    assert body["verification_passed"] is False
    assert body["payload_released"] is False
    joined = " ".join(body["failures"])
    assert "UW-9.9" in joined            # fabricated policy caught
    assert "hallucinated" in joined      # tampered SHAP caught
    assert "credit limit" in joined      # inflated limit caught


def test_recourse_never_cites_protected_attributes(client):
    body = client.post("/api/v1/underwrite", json=APPLICANT).json()
    recourse = body["recourse"]
    forbidden = {"DAYS_BIRTH", "CODE_GENDER", "NAME_FAMILY_STATUS",
                 "DEF_30_CNT_SOCIAL_CIRCLE", "REGION_RATING_CLIENT"}
    assert recourse.get("feature") not in forbidden
    for term in ("age", "gender", "married"):
        assert term not in recourse["advice"].lower()


def test_missing_notes_is_rejected(client):
    payload = dict(APPLICANT, underwriter_notes="")
    r = client.post("/api/v1/underwrite", json=payload)
    assert r.status_code == 422
