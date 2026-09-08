"""
Retrieval helper for scripts/fetch_news.py's SQLite news index. Not a training
component -- this hands relevant recent articles to the model as context at
answer-time (RAG), so it can discuss current events without ever needing its
weights retrained. Graceful by design: returns "" if the index doesn't exist
yet or nothing matches, so callers can always just prepend the result.
"""

import re
import sqlite3
from pathlib import Path

DB_PATH = Path("data/news.db")
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "who", "when",
    "where", "why", "how", "did", "does", "do", "in", "on", "at", "to",
    "of", "for", "and", "or", "with", "about", "tell", "me", "please",
}


def _query_terms(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def get_relevant_context(query: str, limit: int = 3) -> str:
    if not DB_PATH.exists():
        return ""

    terms = _query_terms(query)
    if not terms:
        return ""

    fts_query = " OR ".join(terms)
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            """
            SELECT a.source, a.title, a.summary FROM articles a
            JOIN articles_fts f ON a.rowid = f.rowid
            WHERE articles_fts MATCH ?
            ORDER BY rank LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        conn.close()
    except sqlite3.OperationalError:
        return ""  # index missing/corrupt/empty FTS table -- fail quiet, not loud

    if not rows:
        return ""

    lines = [
        "Your training data is outdated for recent events, and you have no reliable "
        "internal knowledge of them. Answer using ONLY the articles below. If they "
        "don't contain enough to answer, say so directly instead of guessing from "
        "memory. Recent articles:"
    ]
    for source, title, summary in rows:
        snippet = (summary or "")[:280]
        lines.append(f"- [{source}] {title}: {snippet}")
    return "\n".join(lines)
