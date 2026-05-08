"""
collect_data.py
---------------
Collects Reddit posts and comments from r/soccer and r/worldcup
using the PRAW Reddit API wrapper.

Requires environment variables (or .env file):
  REDDIT_CLIENT_ID
  REDDIT_CLIENT_SECRET
  REDDIT_USER_AGENT
"""

import os
import csv
import sqlite3
import logging
from datetime import datetime

import praw
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SUBREDDITS = ["soccer", "worldcup"]
SEARCH_QUERY = "World Cup"
POST_LIMIT = 100
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "../data/raw/reddit_data.csv")
OUTPUT_DB = os.path.join(os.path.dirname(__file__), "../data/raw/reddit_data.db")


def get_reddit_client() -> praw.Reddit:
    """Initialise and return a read-only PRAW Reddit client."""
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "WorldCupSentimentBot/1.0")

    if not client_id or not client_secret:
        raise EnvironmentError(
            "Missing Reddit API credentials. "
            "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables."
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        read_only=True,
    )


def collect_posts(reddit: praw.Reddit) -> list[dict]:
    """Collect posts and top-level comments from target subreddits."""
    records = []

    for subreddit_name in SUBREDDITS:
        logger.info("Collecting from r/%s …", subreddit_name)
        try:
            subreddit = reddit.subreddit(subreddit_name)
            for submission in subreddit.search(SEARCH_QUERY, limit=POST_LIMIT, sort="new"):
                # Store the post title itself
                records.append(
                    {
                        "id": submission.id,
                        "type": "post",
                        "text": submission.title,
                        "score": submission.score,
                        "timestamp": datetime.utcfromtimestamp(submission.created_utc).isoformat(),
                        "subreddit": subreddit_name,
                        "url": submission.url,
                    }
                )

                # Collect top-level comments
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list():
                    if not comment.body or comment.body in ("[deleted]", "[removed]"):
                        continue
                    records.append(
                        {
                            "id": comment.id,
                            "type": "comment",
                            "text": comment.body,
                            "score": comment.score,
                            "timestamp": datetime.utcfromtimestamp(comment.created_utc).isoformat(),
                            "subreddit": subreddit_name,
                            "url": f"https://www.reddit.com{comment.permalink}",
                        }
                    )
        except Exception as exc:
            logger.error("Error collecting from r/%s: %s", subreddit_name, exc)

    logger.info("Collected %d records total.", len(records))
    return records


def save_to_csv(records: list[dict], path: str) -> None:
    """Persist records to a CSV file."""
    if not records:
        logger.warning("No records to save.")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["id", "type", "text", "score", "timestamp", "subreddit", "url"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    logger.info("Saved %d records to %s", len(records), path)


def save_to_sqlite(records: list[dict], path: str) -> None:
    """Persist records to a SQLite database."""
    if not records:
        logger.warning("No records to save.")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reddit_posts (
            id          TEXT PRIMARY KEY,
            type        TEXT,
            text        TEXT,
            score       INTEGER,
            timestamp   TEXT,
            subreddit   TEXT,
            url         TEXT
        )
        """
    )

    cursor.executemany(
        """
        INSERT OR IGNORE INTO reddit_posts (id, type, text, score, timestamp, subreddit, url)
        VALUES (:id, :type, :text, :score, :timestamp, :subreddit, :url)
        """,
        records,
    )

    conn.commit()
    conn.close()
    logger.info("Saved %d records to %s", len(records), path)


def main() -> None:
    """Entry point: collect data and persist to CSV + SQLite."""
    try:
        reddit = get_reddit_client()
        records = collect_posts(reddit)
        save_to_csv(records, OUTPUT_CSV)
        save_to_sqlite(records, OUTPUT_DB)
    except EnvironmentError as exc:
        logger.error("Configuration error: %s", exc)
        raise
    except Exception as exc:
        logger.error("Unexpected error during data collection: %s", exc)
        raise


if __name__ == "__main__":
    main()
