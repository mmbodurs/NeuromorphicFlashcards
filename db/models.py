"""
Database Layer — SQLite
========================
Schema:
    decks       — Named collections of cards
    cards       — Flashcard content (front/back/tags)
    snn_state   — Per-card SNN neuron state (weight, membrane potential, timestamps)
    reviews     — Full review history for analytics

All timestamps are Unix seconds (float).
"""

import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

DB_PATH = Path.home() / ".neuromorphic_flashcards" / "cards.db"


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (and create if needed) the SQLite database."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize_db(conn: sqlite3.Connection):
    """Create all tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS decks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL UNIQUE,
            description TEXT DEFAULT '',
            created_at  REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cards (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            deck_id     INTEGER NOT NULL REFERENCES decks(id) ON DELETE CASCADE,
            front       TEXT NOT NULL,
            back        TEXT NOT NULL,
            tags        TEXT DEFAULT '',
            created_at  REAL NOT NULL,
            is_active   INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS snn_state (
            card_id     INTEGER PRIMARY KEY REFERENCES cards(id) ON DELETE CASCADE,
            weight      REAL NOT NULL DEFAULT 0.05,
            v_mem       REAL NOT NULL DEFAULT 0.0,
            t_last      REAL NOT NULL,
            review_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS reviews (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            card_id     INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
            reviewed_at REAL NOT NULL,
            rating      INTEGER NOT NULL,
            weight_before REAL NOT NULL,
            weight_after  REAL NOT NULL,
            interval_hours REAL NOT NULL
        );
    """)
    conn.commit()


# ------------------------------------------------------------------
# Deck operations
# ------------------------------------------------------------------

def create_deck(conn: sqlite3.Connection, name: str, description: str = "") -> int:
    cur = conn.execute(
        "INSERT INTO decks (name, description, created_at) VALUES (?, ?, ?)",
        (name, description, time.time())
    )
    conn.commit()
    return cur.lastrowid


def list_decks(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("""
        SELECT d.*, COUNT(c.id) as card_count
        FROM decks d
        LEFT JOIN cards c ON c.deck_id = d.id AND c.is_active = 1
        GROUP BY d.id
        ORDER BY d.name
    """).fetchall()


def delete_deck(conn: sqlite3.Connection, deck_id: int):
    conn.execute("DELETE FROM decks WHERE id = ?", (deck_id,))
    conn.commit()


# ------------------------------------------------------------------
# Card operations
# ------------------------------------------------------------------

def create_card(
    conn: sqlite3.Connection,
    deck_id: int,
    front: str,
    back: str,
    tags: str = "",
) -> int:
    now = time.time()
    cur = conn.execute(
        "INSERT INTO cards (deck_id, front, back, tags, created_at) VALUES (?, ?, ?, ?, ?)",
        (deck_id, front.strip(), back.strip(), tags.strip(), now)
    )
    card_id = cur.lastrowid
    # Initialize SNN state for this card
    conn.execute(
        "INSERT INTO snn_state (card_id, weight, v_mem, t_last, review_count) VALUES (?, ?, ?, ?, ?)",
        (card_id, 0.05, 0.0, now, 0)
    )
    conn.commit()
    return card_id


def get_cards_in_deck(conn: sqlite3.Connection, deck_id: int) -> list[sqlite3.Row]:
    return conn.execute("""
        SELECT c.*, s.weight, s.v_mem, s.t_last, s.review_count
        FROM cards c
        JOIN snn_state s ON s.card_id = c.id
        WHERE c.deck_id = ? AND c.is_active = 1
        ORDER BY c.created_at DESC
    """, (deck_id,)).fetchall()


def update_card(conn: sqlite3.Connection, card_id: int, front: str, back: str, tags: str = ""):
    conn.execute(
        "UPDATE cards SET front = ?, back = ?, tags = ? WHERE id = ?",
        (front.strip(), back.strip(), tags.strip(), card_id)
    )
    conn.commit()


def delete_card(conn: sqlite3.Connection, card_id: int):
    conn.execute("UPDATE cards SET is_active = 0 WHERE id = ?", (card_id,))
    conn.commit()


# ------------------------------------------------------------------
# SNN state operations
# ------------------------------------------------------------------

def get_snn_state(conn: sqlite3.Connection, card_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM snn_state WHERE card_id = ?", (card_id,)
    ).fetchone()


def update_snn_state(
    conn: sqlite3.Connection,
    card_id: int,
    weight: float,
    v_mem: float,
    t_last: float,
):
    conn.execute(
        """UPDATE snn_state
           SET weight = ?, v_mem = ?, t_last = ?,
               review_count = review_count + 1
           WHERE card_id = ?""",
        (weight, v_mem, t_last, card_id)
    )
    conn.commit()


def get_due_cards(conn: sqlite3.Connection, deck_id: int, t_now: float | None = None) -> list[sqlite3.Row]:
    """Return all active cards in a deck that are due for review."""
    if t_now is None:
        t_now = time.time()
    return conn.execute("""
        SELECT c.*, s.weight, s.v_mem, s.t_last, s.review_count
        FROM cards c
        JOIN snn_state s ON s.card_id = c.id
        WHERE c.deck_id = ? AND c.is_active = 1
          AND (
            s.review_count = 0
            OR (s.t_last + (4.0 * exp(s.weight * 5.19)) * 3600) <= ?
          )
        ORDER BY s.weight ASC, s.t_last ASC
    """, (deck_id, t_now)).fetchall()


# ------------------------------------------------------------------
# Review history
# ------------------------------------------------------------------

def log_review(
    conn: sqlite3.Connection,
    card_id: int,
    rating: int,
    weight_before: float,
    weight_after: float,
    interval_hours: float,
):
    conn.execute(
        """INSERT INTO reviews
           (card_id, reviewed_at, rating, weight_before, weight_after, interval_hours)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (card_id, time.time(), rating, weight_before, weight_after, interval_hours)
    )
    conn.commit()


# ------------------------------------------------------------------
# Lateral synapse helpers — [[link]] parsing
# ------------------------------------------------------------------

def get_link_refs(text: str) -> list[str]:
    """
    Parse all [[card front]] references from a piece of card text.

    Convention: [[Card B front]] inside Card A's text means
    "A depends on B as a prerequisite." When B is reviewed,
    A receives lateral excitatory current.

    Example:
        get_link_refs("This builds on [[LIF neuron]] and [[STDP]].")
        → ["LIF neuron", "STDP"]
    """
    return re.findall(r'\[\[([^\]]+)\]\]', text)


def get_cards_primed_by(
    conn: sqlite3.Connection,
    source_card_id: int,
    deck_id: int,
) -> list[sqlite3.Row]:
    """
    Find all cards in `deck_id` that contain [[source_front]] in their text.
    These are the cards whose neurons receive lateral current when the source card fires.

    The direction: source card B was reviewed → cards that mention [[B]] are primed.
    """
    source = conn.execute(
        "SELECT front FROM cards WHERE id = ?", (source_card_id,)
    ).fetchone()
    if not source:
        return []

    # Build the LIKE pattern: %[[exact front text]]%
    # Escape % and _ since they are LIKE wildcards in SQLite
    front = source["front"].strip()
    escaped = front.replace("%", r"\%").replace("_", r"\_")
    pattern = f"%[[{escaped}]]%"

    return conn.execute("""
        SELECT c.*, s.weight, s.v_mem, s.t_last, s.review_count
        FROM cards c
        JOIN snn_state s ON s.card_id = c.id
        WHERE c.deck_id = ? AND c.is_active = 1 AND c.id != ?
          AND (c.front LIKE ? ESCAPE '\\' OR c.back LIKE ? ESCAPE '\\')
    """, (deck_id, source_card_id, pattern, pattern)).fetchall()


def get_review_history(conn: sqlite3.Connection, limit: int = 200) -> list[sqlite3.Row]:
    return conn.execute("""
        SELECT r.*, c.front, c.deck_id, d.name as deck_name
        FROM reviews r
        JOIN cards c ON c.id = r.card_id
        JOIN decks d ON d.id = c.deck_id
        ORDER BY r.reviewed_at DESC
        LIMIT ?
    """, (limit,)).fetchall()


def get_deck_stats(conn: sqlite3.Connection, deck_id: int) -> dict:
    """Aggregate statistics for the stats dashboard."""
    total = conn.execute(
        "SELECT COUNT(*) FROM cards WHERE deck_id = ? AND is_active = 1", (deck_id,)
    ).fetchone()[0]

    reviewed = conn.execute("""
        SELECT COUNT(*) FROM cards c
        JOIN snn_state s ON s.card_id = c.id
        WHERE c.deck_id = ? AND c.is_active = 1 AND s.review_count > 0
    """, (deck_id,)).fetchone()[0]

    avg_weight = conn.execute("""
        SELECT AVG(s.weight) FROM snn_state s
        JOIN cards c ON c.id = s.card_id
        WHERE c.deck_id = ? AND c.is_active = 1
    """, (deck_id,)).fetchone()[0] or 0.0

    rating_dist = conn.execute("""
        SELECT rating, COUNT(*) as cnt FROM reviews r
        JOIN cards c ON c.id = r.card_id
        WHERE c.deck_id = ?
        GROUP BY rating
    """, (deck_id,)).fetchall()

    return {
        "total": total,
        "reviewed": reviewed,
        "new": total - reviewed,
        "avg_weight": avg_weight,
        "rating_dist": {row["rating"]: row["cnt"] for row in rating_dist},
    }
