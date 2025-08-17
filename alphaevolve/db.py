import sqlite3
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .config import Config
import math

@dataclass
class ProgramRecord:
    """Represents a program record in the database."""
    code: str
    explanation: str
    score: Optional[float] = None
    gen: Optional[int] = None
    parent_id: Optional[int] = None
    failure_type: Optional[str] = None
    error_message: Optional[str] = None
    id: Optional[int] = None
    experiment_id: Optional[int] = None
    retry_count: int = 0
    total_evaluation_time: float = 0.0
    generation_time: float = 0.0
    total_llm_time: float = 0.0
    total_tokens: int = 0
    conversation: Optional[str] = None  # JSON string containing full LLM conversation
    evaluation_logs: Optional[str] = None  # String containing evaluation logs
    feedback: Optional[str] = None  # String containing LLM-generated feedback
    used_model: Optional[str] = None  # Name of the LLM model used for generation
    
    def to_insert_tuple(self, experiment_id: int) -> Tuple:
        """Convert to tuple for database insertion."""
        return (
            self.code, self.explanation, self.score, self.gen, 
            self.parent_id, experiment_id, self.failure_type, self.error_message, self.retry_count, 
            self.total_evaluation_time, self.generation_time, self.total_llm_time, self.total_tokens,
            self.conversation, self.evaluation_logs, self.feedback, self.used_model
        )


class EvolutionaryDatabase:
    def __init__(self, cfg: Config) -> None:
        # SQLite URI: sqlite:///path/to.db → strip the prefix
        path = cfg.db_uri.replace("sqlite:///", "", 1)
        self.db_path = path
        self.cfg = cfg
        with self.get_connection() as conn:
            self._ensure_schema(conn)
            self.experiment_id = self._ensure_experiment(conn, cfg.exp.label, cfg.exp.notes)

    def get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self, conn) -> None:
        cur = conn.cursor()
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
                explanation TEXT,
                code TEXT,
                score REAL,
                gen  INTEGER NOT NULL,
                parent_id INTEGER,
                experiment_id INTEGER NOT NULL,
                failure_type TEXT,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                total_evaluation_time REAL,
                generation_time REAL,
                total_llm_time REAL,
                total_tokens INTEGER,
                conversation TEXT,
                evaluation_logs TEXT,
                feedback TEXT,
                used_model TEXT,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                     ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_program_exp
                ON programs (experiment_id);
            """
        )

    def _ensure_experiment(self, conn, label: str, notes: str) -> int:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO experiments (label, notes)
            VALUES (?, ?)
            ON CONFLICT(label) DO UPDATE SET notes = excluded.notes
            """,
            (label, notes),
        )
        return cur.execute("SELECT id FROM experiments WHERE label = ?", (label,)).fetchone()[0]

    def add(self, record: ProgramRecord) -> int:
        """
        Add a program to the database.

        Parameters
        ----------
        record : ProgramRecord
            The program record to add

        Returns
        -------
        id : int
            The ID of the inserted program.
        """
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO programs (code, explanation, score, gen, parent_id, experiment_id, failure_type, error_message, retry_count, total_evaluation_time, generation_time, total_llm_time, total_tokens, conversation, evaluation_logs, feedback, used_model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                record.to_insert_tuple(self.experiment_id),
            )
            conn.commit()
            lastrowid = cur.lastrowid
            if lastrowid is None:
                raise RuntimeError("Failed to insert program: lastrowid is None")
            return lastrowid

    def get_latest_generation(self, conn: sqlite3.Connection) -> int:
        """
        Get the latest generation number (highest generation in the database).
        
        Parameters
        ----------
        conn : sqlite3.Connection
            Database connection to use for queries.
        
        Returns
        -------
        latest_gen : int
            The latest generation number, or 0 if no programs exist.
        """
        cur = conn.cursor()
        cur.execute(
            """
            SELECT MAX(gen) as max_gen FROM programs
            WHERE experiment_id = ?
            """,
            (self.experiment_id,),
        )
        result = cur.fetchone()
        return result[0] if result[0] is not None else 0

    def _boltzmann_select_from_pool(self, programs: List[dict], n: int, exclude_id: Optional[int] = None) -> List[dict]:
        """
        Select multiple programs using Boltzmann selection from a given pool of programs.
        
        The probability of selecting a program is proportional to exp(score / temperature).
        Higher scores and lower temperatures make selection more deterministic.
        
        Parameters
        ----------
        programs : List[dict]
            Pool of programs to select from.
        n : int
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
            If no programs exist in the pool.
        """
        if not programs:
            raise RuntimeError("No programs in the provided pool")
        
        # Filter out excluded program if specified
        if exclude_id is not None:
            programs = [p for p in programs if p["id"] != exclude_id]
            
        # If no programs remain after filtering, return empty list
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
        selected = random.choices(programs, weights=probabilities, k=min(n, len(programs)))
        return selected

    def _boltzmann_select_parent_and_inspirations(self, conn, num_inspirations: int) -> Tuple[dict, List[dict]]:
        """
        Select a parent and inspirations using Boltzmann selection from the current generation.
        
        Parameters
        ----------
        conn : sqlite3.Connection
            Database connection to use for queries.
        num_inspirations : int
            Number of inspirations to select.
            
        Returns
        -------
        parent : dict
            Selected parent program.
        inspirations : List[dict]
            Selected inspiration programs (excluding parent).
        """
        # Get current generation
        current_gen = self.get_latest_generation(conn)
        
        # Get all programs from the current generation for this experiment
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM programs
            WHERE experiment_id = ? AND gen = ? AND failure_type IS NULL
            """,
            (self.experiment_id, current_gen),
        )
        programs = [dict(r) for r in cur.fetchall()]
        
        if not programs:
            raise RuntimeError(f"No programs in generation {current_gen}")
        
        # Select parent and inspirations from the same pool
        parent = self._boltzmann_select_from_pool(programs, 1)[0]
        inspirations = self._boltzmann_select_from_pool(programs, num_inspirations, exclude_id=parent["id"])
        
        return parent, inspirations

    def top_k(self, conn: sqlite3.Connection, k: int) -> List[dict]:
        """
        Get the top k programs by score from all generations.

        Parameters
        ----------
        k : int
            Number of top programs to return.
        conn : sqlite3.Connection
            Database connection to use for queries.

        Returns
        -------
        programs : List[dict]
            The top k programs by score from all generations.
        """
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM programs
            WHERE experiment_id = ? AND failure_type IS NULL
            ORDER BY score DESC
            LIMIT ?
            """,
            (self.experiment_id, k),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_recent_generations_programs(self, conn: sqlite3.Connection, num_generations: int, exclude_ids: set = None) -> List[dict]:
        """
        Get programs from the most recent num_generations, excluding specified IDs.
        
        Parameters
        ----------
        conn : sqlite3.Connection
            Database connection to use for queries.
        num_generations : int
            Number of recent generations to consider
        exclude_ids : set, optional
            Set of program IDs to exclude
            
        Returns
        -------
        programs : List[dict]
            Programs from recent generations
        """
        current_gen = self.get_latest_generation(conn)
        start_gen = max(1, current_gen - num_generations + 1)
        
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM programs
            WHERE experiment_id = ? AND gen >= ? AND gen <= ? AND failure_type IS NULL
            ORDER BY gen DESC, score DESC
            """,
            (self.experiment_id, start_gen, current_gen),
        )
        programs = [dict(r) for r in cur.fetchall()]
        
        if exclude_ids:
            programs = [p for p in programs if p["id"] not in exclude_ids]
            
        return programs

    def get_percentile_programs(self, programs: List[dict], percentile: float) -> List[dict]:
        """
        Filter programs to only include those above and including the specified percentile by score.
        
        Parameters
        ----------
        programs : List[dict]
            List of programs to filter
        percentile : float
            Percentile threshold (0-100)
            
        Returns
        -------
        filtered_programs : List[dict]
            Programs above the percentile threshold
        """
        if not programs:
            return []
            
        # Sort by score in ascending order
        sorted_programs = sorted(programs, key=lambda x: x["score"])
        
        # Calculate the index for the percentile
        percentile_index = int(len(sorted_programs) * (percentile / 100.0))
        
        # Return programs above and including the percentile
        return sorted_programs[percentile_index:]

    def random_n(self, conn: sqlite3.Connection, n: int) -> List[dict]:
        """
        Get n random programs.

        Parameters
        ----------
        conn : sqlite3.Connection
            Database connection to use for queries.
        n : int
            Number of random programs to return.

        Returns
        -------
        programs : List[dict]
            n random programs from the current generation.
        """
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM programs
            WHERE experiment_id = ? AND failure_type IS NULL
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (self.experiment_id, n),
        )
        return [dict(r) for r in cur.fetchall()]

    def sample(self) -> Tuple[dict, List[dict]]:
        """
        Pick a single parent row to mutate and a list of inspiration rows
        for prompt context. Parent and inspirations are selected using the specified method.

        Returns
        -------
        parent : dict
            Row chosen as the direct target of mutation.
        inspirations : List[dict]
            Additional rows (≠ parent) to include in the prompt.
        """
        with self.get_connection() as conn:
            selection_method = self.cfg.evolution.selection_method
            num_inspirations = self.cfg.evolution.inspiration_count
            
            if selection_method == "boltzmann":
                # Use Boltzmann selection for both parent and inspirations from current generation
                parent, inspirations = self._boltzmann_select_parent_and_inspirations(conn, num_inspirations)
            elif selection_method == "top_k_and_random":
                # Use a mix of top-k and random selection
                parent, inspirations = self._top_k_and_random_select_parent_and_inspirations(conn, num_inspirations)
            elif selection_method == "enhanced_inspiration":
                # Enhanced inspiration selection from combined pool
                parent, inspirations = self._enhanced_inspiration_select_parent_and_inspirations(conn, num_inspirations)
            else:
                raise ValueError(f"Invalid selection method: {selection_method}")

            return parent, inspirations

    def _top_k_and_random_select_parent_and_inspirations(self, conn: sqlite3.Connection, num_inspirations: int) -> Tuple[dict, List[dict]]:
        """
        Select a parent and inspirations using a mix of top-k and random selection.
        
        Parameters
        ----------
        conn : sqlite3.Connection
            Database connection to use for queries.
        num_inspirations : int
            Number of inspirations to select.
            
        Returns
        -------
        parent : dict
            Selected parent program.
        inspirations : List[dict]
            Selected inspiration programs (excluding parent).
        """
        # Select parent from top-k programs
        top_k_programs = self.top_k(conn, num_inspirations + 1)
        if not top_k_programs:
            raise RuntimeError("No programs available for selection")
        
        parent = random.choice(top_k_programs)
        
        # Select inspirations using the original top_k_and_random logic
        half = num_inspirations // 2
        
        # a) strongest half (top-k but skip parent)
        strong = [row for row in top_k_programs if row["id"] != parent["id"]][:half]

        # b) diverse half (uniform random, also skip parent & already-chosen rows)
        diverse_pool = [row for row in self.random_n(conn, num_inspirations * 2)
                        if row["id"] != parent["id"]
                        and row["id"] not in {r["id"] for r in strong}]
        diverse = random.sample(diverse_pool, min(num_inspirations - len(strong), len(diverse_pool)))

        inspirations = strong + diverse
        random.shuffle(inspirations)   # avoid positional bias in the prompt
        
        return parent, inspirations


    def _enhanced_inspiration_select_parent_and_inspirations(self, conn: sqlite3.Connection, num_inspirations: int) -> Tuple[dict, List[dict]]:
        """
        Select a parent and inspirations using Boltzmann sampling from a combined pool:
        - Combines top-k all-time pool and recent generations above percentile threshold
        - Uses Boltzmann selection from the combined pool for both parent and inspirations
        - Fallback to random selection if combined pool is insufficient
        
        Parameters
        ----------
        conn : sqlite3.Connection
            Database connection to use for queries.
        num_inspirations : int
            Number of inspirations to select
            
        Returns
        -------
        parent : dict
            Selected parent program.
        inspirations : List[dict]
            Selected inspiration programs (excluding parent).
        """
        # Elites pool is the top-k all-time pool, default is double the population
        k = self.cfg.evolution.population_size * 2
        
        # Get top-k all-time programs
        top_k_pool = self.top_k(conn, k + 1)
        
        # Get recent generations programs (excluding top-k pool)
        recent_generations = self.cfg.evolution.recent_generations
        recent_percentile = self.cfg.evolution.recent_percentile
        
        exclude_ids = {p["id"] for p in top_k_pool}
        recent_programs = self.get_recent_generations_programs(conn, recent_generations, exclude_ids)
        
        # Filter recent programs by percentile
        filtered_recent_pool = self.get_percentile_programs(recent_programs, recent_percentile)
        
        # Combine the pools
        combined_pool = top_k_pool + filtered_recent_pool
        
        # Use Boltzmann selection from the combined pool for both parent and inspirations
        try:
            # Select parent first
            parent = self._boltzmann_select_from_pool(combined_pool, 1)[0]
            # Then select inspirations, excluding the parent
            inspirations = self._boltzmann_select_from_pool(combined_pool, num_inspirations, exclude_id=parent["id"])
            # print(f"Selected parent and {len(inspirations)} inspirations using Boltzmann selection from combined pool (top-k: {len(top_k_pool)}, recent: {len(filtered_recent_pool)})")
        except RuntimeError:
            # Fallback to random selection if combined pool is insufficient
            print(f"Combined pool insufficient for parent and {num_inspirations} inspirations. Falling back to random selection.")
            all_programs = self.random_n(conn, num_inspirations + 1)
            if not all_programs:
                raise RuntimeError("No programs available for selection")
            parent = all_programs[0]
            inspirations = all_programs[1:]
        
        # Shuffle to avoid positional bias in the prompt
        random.shuffle(inspirations)
        return parent, inspirations