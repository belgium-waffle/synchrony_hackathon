"""Embed the policy corpus and upsert it into PostgreSQL.

    python -m scripts.seed_policies

Idempotent: re-running re-embeds and overwrites, which is what you want after
editing a policy body or switching embedding models.
"""
from __future__ import annotations

import logging
import sys

from sqlalchemy import text

from app.database import init_engine, session_scope
from app.policies import SEED_POLICIES, Embedder, to_pgvector_literal

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger("seed")

UPSERT = text("""
    INSERT INTO underwriting_policies (policy_id, title, body, embedding, updated_at)
    VALUES (:policy_id, :title, :body, CAST(:embedding AS vector), now())
    ON CONFLICT (policy_id) DO UPDATE SET
        title = EXCLUDED.title,
        body = EXCLUDED.body,
        embedding = EXCLUDED.embedding,
        updated_at = now()
""")


def main() -> int:
    if init_engine() is None:
        log.error("DATABASE_URL is not set or PostgreSQL is unreachable. Nothing seeded.")
        return 1

    embedder = Embedder()
    if embedder.backend_name == "hashing-fallback":
        log.error("Refusing to seed with the hashing fallback embedder: the vectors would "
                  "not match what a MiniLM-backed service computes at query time. "
                  "Install sentence-transformers and retry.")
        return 2

    corpus = [f"{p['title']}. {p['body']}" for p in SEED_POLICIES]
    vectors = embedder.encode_many(corpus)

    with session_scope() as s:
        for policy, vec in zip(SEED_POLICIES, vectors):
            s.execute(UPSERT, {**policy, "embedding": to_pgvector_literal(vec)})
        count = s.execute(text("SELECT COUNT(*) FROM underwriting_policies")).scalar()

    log.info("Seeded %d policies (%d rows in table).", len(SEED_POLICIES), count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
