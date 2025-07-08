import sqlite3
import random
from typing import List, Tuple, Optional
from .config import Config

class EvolutionaryDatabase:
    def __init__(self, cfg: Config) -> None:
        # SQLite URI: sqlite:///path/to.db → strip the prefix
        path = cfg.db_uri.replace("sqlite:///", "", 1)
        self.conn = sqlite3.connect(path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row  # enable dict-like access to rows
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema()
        self.cfg = cfg
        self.experiment_id = self._ensure_experiment(
            cfg.exp.label, cfg.exp.notes
        )

    def _ensure_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT UNIQUE NOT NULL,
                notes TEXT,
                started_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS programs (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                score REAL NOT NULL,
                gen  INTEGER NOT NULL,
                parent_id INTEGER,
                experiment_id INTEGER NOT NULL,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                     ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_program_exp
                ON programs (experiment_id);
            """
        )

    def _ensure_experiment(self, label: str, notes: str) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO experiments (label, notes)
            VALUES (?, ?)
            ON CONFLICT(label) DO UPDATE SET notes = excluded.notes
            """,
            (label, notes),
        )
        return cur.execute("SELECT id FROM experiments WHERE label = ?", (label,)).fetchone()[0]

    def add(self, code: str, score: float, gen: int, parent_id: Optional[int]) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO programs (code, score, gen, parent_id, experiment_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (code, score, gen, parent_id, self.experiment_id),
        )
        return cur.lastrowid

    def top_k(self, k: int):
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM programs
            WHERE experiment_id = ?
            ORDER BY score DESC
            LIMIT ?
            """,
            (self.experiment_id, k),
        )
        return [dict(r) for r in cur.fetchall()]

    def random_n(self, n: int):
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM programs
            WHERE experiment_id = ?
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (self.experiment_id, n),
        )
        return [dict(r) for r in cur.fetchall()]
    
    def sample(self) -> Tuple[dict, List[dict]]:
        """
        Pick a single parent row to mutate and a list of inspiration rows
        for prompt context.

        Returns
        -------
        parent : dict
            Row chosen as the direct target of mutation.
        inspirations : List[dict]
            Additional rows (≠ parent) to include in the prompt.
        """
        # 1. ----- parent selection ------------------------------------------
        try:
            parent = self._boltzmann_select()
        except RuntimeError:           # archive empty ⇒ caller should seed first row
            raise
        except Exception:              # safety net – fall back to pure random choice
            parent = self.random_n(1)[0]

        # 2. ----- inspiration selection -------------------------------------
        k = self.cfg.evolution.inspiration_count
        half = k // 2

        # a) strongest half (top-k but skip parent)
        strong = [row for row in self.top_k(k + 1) if row["id"] != parent["id"]][:half]

        # b) diverse half (uniform random, also skip parent & already-chosen rows)
        diverse_pool = [row for row in self.random_n(k * 2)
                        if row["id"] != parent["id"]
                        and row["id"] not in {r["id"] for r in strong}]
        diverse = random.sample(diverse_pool, min(k - len(strong), len(diverse_pool)))

        inspirations = strong + diverse
        random.shuffle(inspirations)   # avoid positional bias in the prompt

        return parent, inspirations
