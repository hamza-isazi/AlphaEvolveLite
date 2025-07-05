import psycopg2
import math
import random
from psycopg2.extras import RealDictCursor
from typing import List, Tuple, Optional
from .config import EvolCfg

class EvolutionaryDatabase:
    def __init__(self, db_uri: str, evol_cfg: EvolCfg) -> None:
        self.conn = psycopg2.connect(db_uri, cursor_factory=RealDictCursor)
        self._ensure_schema()
        self.evol_cfg = evol_cfg

    def _ensure_schema(self) -> None:
        with self.conn, self.conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS programs (
                    id SERIAL PRIMARY KEY,
                    code TEXT NOT NULL,
                    score DOUBLE PRECISION NOT NULL,
                    gen  INT NOT NULL,
                    parent_id INT
                );
                """
            )

    def add(self, code: str, score: float, gen: int, parent_id: Optional[int]) -> int:
        with self.conn, self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO programs (code, score, gen, parent_id) VALUES (%s,%s,%s,%s) RETURNING id",
                (code, score, gen, parent_id),
            )
            return cur.fetchone()["id"]

    def _top_k(self, k: int) -> List[dict]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM programs ORDER BY score DESC LIMIT %s", (k,))
            return cur.fetchall()

    def _random_n(self, n: int) -> List[dict]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM programs ORDER BY random() LIMIT %s", (n,))
            return cur.fetchall()

    def _boltzmann_select(self):
        rows = self.database.top_k(self.evol_cfg.population_size)
        if not rows:
            raise RuntimeError("Archive empty")
        probs = [
            math.exp(r["score"] / self.evol_cfg.temperature) for r in rows
        ]
        total = sum(probs)
        probs = [p / total for p in probs]
        return random.choices(rows, probs, k=1)[0]
    
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
            parent = self._random_n(1)[0]

        # 2. ----- inspiration selection -------------------------------------
        k = self.evol_cfg.inspiration_count
        half = k // 2

        # a) strongest half (top-k but skip parent)
        strong = [row for row in self._top_k(k + 1) if row["id"] != parent["id"]][:half]

        # b) diverse half (uniform random, also skip parent & already-chosen rows)
        diverse_pool = [row for row in self._random_n(k * 2)
                        if row["id"] != parent["id"]
                        and row["id"] not in {r["id"] for r in strong}]
        diverse = random.sample(diverse_pool, min(k - len(strong), len(diverse_pool)))

        inspirations = strong + diverse
        random.shuffle(inspirations)   # avoid positional bias in the prompt

        return parent, inspirations