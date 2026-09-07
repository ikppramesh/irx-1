#!/usr/bin/env python3
"""
Fetch Indian news RSS feeds into a local SQLite index with full-text search.
This is retrieval data, not training data -- scripts/news_context.py queries
it at answer-time and hands relevant recent articles to the model as context.
The model's weights never change; only this index does. Meant to run on a
schedule (see scripts/com.irx1.newsfetch.plist) so it always has current
articles without ever needing retraining.

Usage:
    python scripts/fetch_news.py
"""

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
]

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
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        summary = (item.findtext("description") or "").strip()
        if title and link:
            items.append({"title": title, "link": link, "summary": summary})
    return items


def main():
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
    conn.close()


if __name__ == "__main__":
    main()
