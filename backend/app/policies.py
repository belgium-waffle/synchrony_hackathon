"""Stream B — qualitative retrieval over the underwriting policy corpus.

Production path: pgvector cosine search inside PostgreSQL. Development path: the same
cosine search executed in-process over the seed corpus, so the service runs on a laptop
with no database. Both return identical shapes, and the verifier treats them the same.
"""
from __future__ import annotations

import logging
import re
import zlib
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np
from sqlalchemy import text

from .config import get_settings
from .database import get_engine, session_scope

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed corpus. Loaded into PostgreSQL by scripts/seed_policies.py, and used
# directly by the in-memory fallback.
# ---------------------------------------------------------------------------
SEED_POLICIES: List[dict] = [
    {
        "policy_id": "UW-1.1",
        "title": "Thin-file and no-bureau applicants",
        "body": (
            "Applicants with no credit bureau record or fewer than three tradelines may be "
            "underwritten using alternative data, including verified bank inflows, utility "
            "and telecom payment behaviour and length of mobile number tenure. A no-hit "
            "bureau result alone is not grounds for decline."
        ),
    },
    {
        "policy_id": "UW-2.3",
        "title": "Unverifiable or absent employment income",
        "body": (
            "Where employment length is absent or carries a sentinel value, or where income "
            "is self-declared and cannot be evidenced by payslips, the underwriter must "
            "document at least six months of inflow history and rely on the trailing median "
            "rather than the peak. Absence of employment is not on its own an adverse factor "
            "where household repayment capacity is demonstrated."
        ),
    },
    {
        "policy_id": "UW-3.2",
        "title": "Reliance on external scoring sources",
        "body": (
            "External normalised scores may be used as a primary ranking signal. Where two "
            "or more external sources are missing, the application may not be auto-approved "
            "and must be routed to manual review. A single strong external source does not "
            "substitute for a complete file."
        ),
    },
    {
        "policy_id": "UW-4.5",
        "title": "Credit bureau enquiry velocity and social circle defaults",
        "body": (
            "Four or more credit bureau enquiries in the trailing twelve months, or two or "
            "more observed defaults within the applicant's social circle, is treated as an "
            "adverse indicator of credit-seeking behaviour and must be recorded in the "
            "adverse action reasons where it contributes materially to the decision."
        ),
    },
    {
        "policy_id": "UW-5.1",
        "title": "Leverage limits and credit line assignment",
        "body": (
            "An annuity-to-income ratio above 0.45 caps the assigned credit line at one "
            "month of verified income. A ratio above 0.60, or a payment rate inconsistent "
            "with the stated term, requires decline unless compensating factors are "
            "documented in the audit trail."
        ),
    },
]


@dataclass
class RetrievedPolicy:
    policy_id: str
    title: str
    body: str
    similarity: float


# --------------------------------------------------------------- embedder ----
class _HashingEmbedder:
    """Deterministic offline fallback (hashed bag-of-words, L2-normalised).

    Emits the same dimensionality as MiniLM so the pgvector column definition does not
    change between modes. Retrieval quality is lower; the contract is identical.
    """

    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, texts, normalize_embeddings: bool = True, **_: Any) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        mat = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", str(t).lower()):
                if len(tok) >= 3:
                    mat[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
        if normalize_embeddings:
            mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        return mat


class Embedder:
    """Wraps sentence-transformers with a graceful offline fallback."""

    def __init__(self):
        settings = get_settings()
        self.dim = settings.EMBEDDING_DIM
        self.backend_name = settings.EMBEDDING_MODEL
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            loaded_dim = self._model.get_sentence_embedding_dimension()
            if loaded_dim != self.dim:
                raise ValueError(
                    f"EMBEDDING_DIM={self.dim} does not match {settings.EMBEDDING_MODEL} "
                    f"({loaded_dim}). Update the setting and the vector() column together."
                )
            log.info("Embedder: %s (dim %d)", settings.EMBEDDING_MODEL, self.dim)
        except Exception as exc:
            log.warning("sentence-transformers unavailable (%s); using hashing embedder.",
                        exc.__class__.__name__)
            self._model = _HashingEmbedder(self.dim)
            self.backend_name = "hashing-fallback"

    def encode_one(self, text_input: str) -> np.ndarray:
        vec = np.asarray(self._model.encode([text_input], normalize_embeddings=True),
                         dtype="float32")[0]
        return vec

    def encode_many(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts), normalize_embeddings=True),
                          dtype="float32")


def to_pgvector_literal(vec: np.ndarray) -> str:
    """pgvector accepts a bracketed literal; cast it server-side with ::vector."""
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


# ------------------------------------------------------------- repository ----
class PolicyRepository:
    """Cosine retrieval over the policy corpus, pgvector-backed where available."""

    def __init__(self, embedder: Optional[Embedder] = None):
        self.embedder = embedder or Embedder()
        self._memory_vectors: Optional[np.ndarray] = None
        self.backend: str = "in-memory"
        self.policy_count: int = 0

    def initialise(self) -> "PolicyRepository":
        if get_engine() is not None:
            try:
                with session_scope() as s:
                    count = s.execute(text("SELECT COUNT(*) FROM underwriting_policies")).scalar()
                if count:
                    self.backend, self.policy_count = "pgvector", int(count)
                    log.info("Policy store: pgvector (%d policies)", count)
                    return self
                log.warning("underwriting_policies is empty — run scripts/seed_policies.py")
            except Exception as exc:
                log.error("pgvector policy table unavailable (%s); using in-memory store.",
                          exc.__class__.__name__)

        corpus = [f"{p['title']}. {p['body']}" for p in SEED_POLICIES]
        self._memory_vectors = self.embedder.encode_many(corpus)
        self.backend, self.policy_count = "in-memory", len(SEED_POLICIES)
        log.info("Policy store: in-memory (%d policies)", self.policy_count)
        return self

    def retrieve(self, notes: str, k: int = 2) -> List[RetrievedPolicy]:
        if not notes or not notes.strip():
            return []
        query = self.embedder.encode_one(notes)
        if self.backend == "pgvector":
            return self._retrieve_pgvector(query, k)
        return self._retrieve_memory(query, k)

    def _retrieve_pgvector(self, query: np.ndarray, k: int) -> List[RetrievedPolicy]:
        # <=> is pgvector's cosine distance; similarity = 1 - distance.
        sql = text("""
            SELECT policy_id, title, body,
                   1 - (embedding <=> CAST(:q AS vector)) AS similarity
            FROM underwriting_policies
            WHERE is_active = TRUE
            ORDER BY embedding <=> CAST(:q AS vector)
            LIMIT :k
        """)
        try:
            with session_scope() as s:
                rows = s.execute(sql, {"q": to_pgvector_literal(query), "k": k}).fetchall()
            return [RetrievedPolicy(r.policy_id, r.title, r.body, round(float(r.similarity), 4))
                    for r in rows]
        except Exception as exc:
            log.error("pgvector retrieval failed (%s); falling back in-place.",
                      exc.__class__.__name__)
            if self._memory_vectors is None:
                corpus = [f"{p['title']}. {p['body']}" for p in SEED_POLICIES]
                self._memory_vectors = self.embedder.encode_many(corpus)
            return self._retrieve_memory(query, k)

    def _retrieve_memory(self, query: np.ndarray, k: int) -> List[RetrievedPolicy]:
        sims = self._memory_vectors @ query          # vectors are L2-normalised
        order = np.argsort(-sims)[:k]
        return [RetrievedPolicy(SEED_POLICIES[i]["policy_id"], SEED_POLICIES[i]["title"],
                                SEED_POLICIES[i]["body"], round(float(sims[i]), 4))
                for i in order]

    def all_policy_ids(self) -> List[str]:
        return [p["policy_id"] for p in SEED_POLICIES]
