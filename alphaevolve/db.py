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
    score: Optional[float]
    gen: int
    parent_id: Optional[int]
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
                explanation TEXT NOT NULL,
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
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO programs (code, explanation, score, gen, parent_id, experiment_id, failure_type, error_message, retry_count, total_evaluation_time, generation_time, total_llm_time, total_tokens, conversation, evaluation_logs, feedback, used_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            record.to_insert_tuple(self.experiment_id),
        )
        lastrowid = cur.lastrowid
        if lastrowid is None:
            raise RuntimeError("Failed to insert program: lastrowid is None")
        return lastrowid

    def get_current_generation(self) -> int:
        """
        Get the current generation number (highest generation in the database).
        
        Returns
        -------
        current_gen : int
            The current generation number, or 0 if no programs exist.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT MAX(gen) as max_gen FROM programs
            WHERE experiment_id = ?
            """,
            (self.experiment_id,),
        )
        result = cur.fetchone()
        return result[0] if result[0] is not None else 0

    def _boltzmann_select(self) -> dict:
        """
        Select a parent program using Boltzmann selection from the most recent generation.
        
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
        # Get current generation
        current_gen = self.get_current_generation()
        
        # Get all programs from the current generation for this experiment
        cur = self.conn.cursor()
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

    def _boltzmann_select_multiple(self, n: int, exclude_id: Optional[int] = None) -> List[dict]:
        """
        Select multiple programs using Boltzmann selection from the most recent generation.
        
        The probability of selecting a program is proportional to exp(score / temperature).
        Higher scores and lower temperatures make selection more deterministic.
        
        Parameters
        ----------
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
            If no programs exist in the database.
        """
        # Get current generation
        current_gen = self.get_current_generation()
        
        # Get all programs from the current generation for this experiment
        cur = self.conn.cursor()
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
        selected = random.choices(programs, weights=probabilities, k=min(n, len(programs)))
        return selected

    def top_k(self, k: int) -> List[dict]:
        """
        Get the top k programs by score from all generations.

        Returns
        -------
        programs : List[dict]
            The top k programs by score from all generations.
        """
        cur = self.conn.cursor()
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

    def get_recent_generations_programs(self, num_generations: int, exclude_ids: set = None) -> List[dict]:
        """
        Get programs from the most recent num_generations, excluding specified IDs.
        
        Parameters
        ----------
        num_generations : int
            Number of recent generations to consider
        exclude_ids : set, optional
            Set of program IDs to exclude
            
        Returns
        -------
        programs : List[dict]
            Programs from recent generations
        """
        current_gen = self.get_current_generation()
        start_gen = max(1, current_gen - num_generations + 1)
        
        cur = self.conn.cursor()
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

    def random_n(self, n: int) -> List[dict]:
        """
        Get n random programs.

        Returns
        -------
        programs : List[dict]
            n random programs from the current generation.
        """
        cur = self.conn.cursor()
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
        num_inspirations = self.cfg.evolution.inspiration_count
        selection_method = self.cfg.evolution.selection_method
        if selection_method == "boltzmann":
            # Use Boltzmann selection for inspirations, excluding the parent
            inspirations = self._boltzmann_select_multiple(num_inspirations, exclude_id=parent["id"])
        elif selection_method == "top_k_and_random": # Use a mix of top-k and random selection
            half = num_inspirations // 2
            # a) strongest half (top-k but skip parent)
            strong = [row for row in self.top_k(num_inspirations + 1) if row["id"] != parent["id"]][:half]

            # b) diverse half (uniform random, also skip parent & already-chosen rows)
            diverse_pool = [row for row in self.random_n(num_inspirations * 2)
                            if row["id"] != parent["id"]
                            and row["id"] not in {r["id"] for r in strong}]
            diverse = random.sample(diverse_pool, min(num_inspirations - len(strong), len(diverse_pool)))

            inspirations = strong + diverse
            random.shuffle(inspirations)   # avoid positional bias in the prompt
        elif selection_method == "enhanced_inspiration":
            # New enhanced inspiration selection method
            inspirations = self._enhanced_inspiration_selection(num_inspirations, parent["id"])
        else:
            raise ValueError(f"Invalid selection method: {selection_method}")

        return parent, inspirations

    def _enhanced_inspiration_selection(self, num_inspirations: int, exclude_parent_id: int) -> List[dict]:
        """
        Enhanced inspiration selection with the new logic:
        - Half from top-k all-time pool
        - Half from recent generations above percentile threshold
        - Fallback to random selection until both pools are >= k
        
        Parameters
        ----------
        num_inspirations : int
            Number of inspirations to select
        exclude_parent_id : int
            ID of parent program to exclude
            
        Returns
        -------
        inspirations : List[dict]
            Selected inspiration programs
        """
        # Elites pool is the top-k all-time pool, default is double the population
        k = self.cfg.evolution.population_size * 2
        half = num_inspirations // 2
        
        # Get top-k all-time programs (excluding parent)
        top_k_pool = [p for p in self.top_k(k + 1) if p["id"] != exclude_parent_id]
        
        # Get recent generations programs (excluding parent and top-k pool)
        recent_generations = self.cfg.evolution.recent_generations
        recent_percentile = self.cfg.evolution.recent_percentile
        
        exclude_ids = {exclude_parent_id} | {p["id"] for p in top_k_pool}
        recent_programs = self.get_recent_generations_programs(recent_generations, exclude_ids)
        
        # Filter recent programs by percentile
        filtered_recent_pool = self.get_percentile_programs(recent_programs, recent_percentile)
        
        # Check if both pools are sufficient size (>= k), if so, select half from each
        if len(top_k_pool) >= k and len(filtered_recent_pool) >= k:
            # Both pools are sufficient - select half from each
            top_k_selected = random.sample(top_k_pool, half)
            recent_selected = random.sample(filtered_recent_pool, half)
            
            inspirations = top_k_selected + recent_selected
        # If the top-k pool is sufficient size, select all inspirations from the top-k pool (top-k pool gets filled first)
        elif len(top_k_pool) >= k:
            inspirations = random.sample(top_k_pool, num_inspirations)
            print(f"Selected {len(inspirations)} inspirations from top-k pool")
        # Otherwise, fall back to full random sampling
        else:
            print(f"Not enough programs in top-k pool or recent generations pool to select {num_inspirations} inspirations. Falling back to full random sampling.")
            inspirations = self.random_n(num_inspirations)
        
        # Shuffle to avoid positional bias in the prompt
        random.shuffle(inspirations)
        return inspirations