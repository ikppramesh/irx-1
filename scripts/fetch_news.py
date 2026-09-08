#!/usr/bin/env python3
"""
Fetch Indian news + AI/tech RSS feeds into a local SQLite index with full-text
search. This is retrieval data, not training data -- scripts/news_context.py
queries it at answer-time and hands relevant recent articles to the model as
context. The model's weights never change; only this index does. Meant to run
on a schedule (see scripts/com.irx1.newsfetch.plist) so it always has current
articles without ever needing retraining -- deliberately not baked into
weights, since AI model releases go stale faster than almost any other fact
category (a fine-tuned "current SOTA model" claim would be wrong within
months, worse than not knowing at all).

Also optionally publishes a compact JSON snapshot (--json-out) for any client
that isn't this Mac -- a mobile app, someone else's laptop, anyone who
downloaded the model elsewhere -- to fetch over plain HTTP and search locally.
See .github/workflows/fetch-news.yml, which runs this on a schedule and
publishes the JSON via GitHub Pages: free, no server to keep alive, works from
anywhere with no local Python/SQLite setup required.

Usage:
    python scripts/fetch_news.py
    python scripts/fetch_news.py --json-out docs/news.json
"""

import argparse
import json
import sqlite3
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

FEEDS = [
    ("Times of India", "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
    ("The Hindu", "https://www.thehindu.com/news/national/feeder/default.rss"),
    ("Indian Express", "https://indianexpress.com/section/india/feed/"),
    ("NDTV", "https://feeds.feedburner.com/ndtvnews-top-stories"),
    ("LiveMint", "https://www.livemint.com/rss/news"),
    ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/"),
    ("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
    ("MIT Technology Review", "https://www.technologyreview.com/feed/"),
]

ATOM_NS = "{http://www.w3.org/2005/Atom}"

DB_PATH = Path("data/news.db")
RETENTION_DAYS = 7
USER_AGENT = "Mozilla/5.0 (compatible; IRx-1-NewsFetch/1.0)"


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            link TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT,
            fetched_at INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
            title, summary, content='articles', content_rowid='rowid'
        )
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS articles_ai AFTER INSERT ON articles BEGIN
            INSERT INTO articles_fts(rowid, title, summary)
            VALUES (new.rowid, new.title, new.summary);
        END
    """)
    conn.commit()


def fetch_feed(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = resp.read()

    root = ET.fromstring(body)
    if root.tag == f"{ATOM_NS}feed":
        return _parse_atom(root)
    return _parse_rss2(root)


def _parse_rss2(root: ET.Element) -> list[dict]:
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        summary = (item.findtext("description") or "").strip()
        if title and link:
            items.append({"title": title, "link": link, "summary": summary})
    return items


def _parse_atom(root: ET.Element) -> list[dict]:
    items = []
    for entry in root.iter(f"{ATOM_NS}entry"):
        title = (entry.findtext(f"{ATOM_NS}title") or "").strip()
        link_el = entry.find(f"{ATOM_NS}link[@rel='alternate']") or entry.find(f"{ATOM_NS}link")
        link = link_el.get("href", "").strip() if link_el is not None else ""
        summary = (
            entry.findtext(f"{ATOM_NS}summary")
            or entry.findtext(f"{ATOM_NS}content")
            or ""
        ).strip()
        if title and link:
            items.append({"title": title, "link": link, "summary": summary})
    return items


def write_json_snapshot(conn: sqlite3.Connection, out_path: Path) -> int:
    rows = conn.execute(
        "SELECT source, title, summary, link, fetched_at FROM articles "
        "ORDER BY fetched_at DESC"
    ).fetchall()
    articles = [
        {"source": s, "title": t, "summary": summ, "link": link, "fetched_at": fa}
        for s, t, summ, link, fa in rows
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {"generated_at": int(time.time()), "count": len(articles), "articles": articles},
            f,
            ensure_ascii=False,
        )
    return len(articles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=None,
                         help="Also write a JSON snapshot to this path (for static hosting)")
    args = parser.parse_args()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    now = int(time.time())
    total_new = 0
    for source, url in FEEDS:
        try:
            items = fetch_feed(url)
        except Exception as e:
            print(f"[{source}] fetch failed: {e}")
            continue

        new_count = 0
        for item in items:
            cur = conn.execute(
                "INSERT OR IGNORE INTO articles (link, source, title, summary, fetched_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (item["link"], source, item["title"], item["summary"], now),
            )
            if cur.rowcount:
                new_count += 1
        conn.commit()
        total_new += new_count
        print(f"[{source}] {len(items)} items, {new_count} new")

    cutoff = now - RETENTION_DAYS * 86400
    deleted = conn.execute("DELETE FROM articles WHERE fetched_at < ?", (cutoff,)).rowcount
    conn.commit()
    conn.execute("INSERT INTO articles_fts(articles_fts) VALUES ('rebuild')")
    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    print(f"Added {total_new} new articles, expired {deleted} (>{RETENTION_DAYS}d old), {total} total in index")

    if args.json_out:
        n = write_json_snapshot(conn, args.json_out)
        print(f"Wrote JSON snapshot ({n} articles) -> {args.json_out}")

    conn.close()


if __name__ == "__main__":
    main()
