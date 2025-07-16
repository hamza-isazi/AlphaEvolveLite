import sqlite3
import random
from typing import List, Tuple, Optional
from .config import Config
import math

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
        """
        Add a program to the database.

        Returns
        -------
        id : int
            The ID of the inserted program.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO programs (code, score, gen, parent_id, experiment_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (code, score, gen, parent_id, self.experiment_id),
        )
        lastrowid = cur.lastrowid
        if lastrowid is None:
            raise RuntimeError("Failed to insert program: lastrowid is None")
        return lastrowid

    def top_k(self, k: int) -> List[dict]:
        """
        Get the top k programs by score.

        Returns
        -------
        programs : List[dict]
            The top k programs by score.
        """
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

    def random_n(self, n: int) -> List[dict]:
        """
        Get n random programs.

        Returns
        -------
        programs : List[dict]
            n random programs.
        """
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

    def _boltzmann_select(self) -> dict:
        """
        Select a parent program using Boltzmann selection.
        
        The probability of selecting a program is proportional to exp(score / temperature).
        Higher scores and lower temperatures make selection more deterministic.
        
        Returns
        -------
        program : dict
            Selected program row.
            
        Raises
        ------
        RuntimeError
            If no programs exist in the database.
        """
        # Get all programs for this experiment
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM programs
            WHERE experiment_id = ?
            """,
            (self.experiment_id,),
        )
        programs = [dict(r) for r in cur.fetchall()]
        
        if not programs:
            raise RuntimeError("No programs in database")
        
        # Calculate Boltzmann weights
        temperature = self.cfg.evolution.temperature
        scores = [program["score"] for program in programs]
        max_score = max(scores)
        weights = [
            math.exp((score - max_score) / temperature)
            for score in scores
        ]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        # Select based on probabilities
        selected = random.choices(programs, weights=probabilities, k=1)[0]
        return selected

    def _boltzmann_select_multiple(self, k: int, exclude_id: Optional[int] = None) -> List[dict]:
        """
        Select multiple programs using Boltzmann selection.
        
        The probability of selecting a program is proportional to exp(score / temperature).
        Higher scores and lower temperatures make selection more deterministic.
        
        Parameters
        ----------
        k : int
            Number of programs to select.
        exclude_id : Optional[int]
            ID of program to exclude from selection.
            
        Returns
        -------
        programs : List[dict]
            Selected program rows.
            
        Raises
        ------
        RuntimeError
            If no programs exist in the database.
        """
        # Get all programs for this experiment
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM programs
            WHERE experiment_id = ?
            """,
            (self.experiment_id,),
        )
        programs = [dict(r) for r in cur.fetchall()]
        
        if not programs:
            raise RuntimeError("No programs in database")
        
        # Filter out excluded program if specified
        if exclude_id is not None:
            programs = [p for p in programs if p["id"] != exclude_id]
            
        # In the first generation, there is only one program, so we need to return an empty list
        if not programs:
            return []
        
        # Calculate Boltzmann weights
        temperature = self.cfg.evolution.temperature
        scores = [program["score"] for program in programs]
        max_score = max(scores)
        weights = [
            math.exp((score - max_score) / temperature)
            for score in scores
        ]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        # Select based on probabilities (without replacement)
        selected = random.choices(programs, weights=probabilities, k=min(k, len(programs)))
        return selected

    def sample(self, selection_method: str = "top_k_and_random") -> Tuple[dict, List[dict]]:
        """
        Pick a single parent row to mutate and a list of inspiration rows
        for prompt context. Inspirations are selected using the specified method.

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
        if selection_method == "boltzmann":
            # Use Boltzmann selection for inspirations, excluding the parent
            inspirations = self._boltzmann_select_multiple(k, exclude_id=parent["id"])
        elif selection_method == "top_k_and_random": # Use a mix of top-k and random selection
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
        else:
            raise ValueError(f"Invalid selection method: {selection_method}")

        return parent, inspirations