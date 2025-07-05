
from pathlib import Path

from .log import init_logger
from .patcher import PatchApplier
from .prompts import PromptSampler
from .problem import Problem
from .db import EvolutionaryDatabase
from .config import Config
from .llm import OpenAIEngine


class EvolutionController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = init_logger()
        self.database = EvolutionaryDatabase(cfg.db_uri, cfg.evolution)
        self.logger.info("Connected to database at %s", cfg.db_uri)
        self.problem = Problem(cfg.problem_entry, cfg.problem_eval)
        self.prompt_sampler = PromptSampler(self.database)
        self.llm = OpenAIEngine(cfg.llm.model)
        self.patcher = PatchApplier()

        # Seed archive with original solution
        seed_code = Path(cfg.problem_entry).read_text()
        seed_score = self.problem.evaluate(cfg.problem_entry)
        self.database.add(seed_code, seed_score, gen=0, parent_id=None)
        self.logger.info("Seed score %.3f", seed_score)

    def run(self):
        for gen in range(1, self.cfg.evolution.max_generations + 1):
            parent_row, inspiration_rows = self.database.sample()
            prompt = self.prompt_sampler.build(parent_row, inspiration_rows)
            diff = self.llm.generate(prompt)
            child_program = self.patcher.apply_diff(parent_row["code"], diff)

            if not child_program or not self.patcher.is_valid(child_program):
                self.logger.info("Gen %d: invalid patch, skipping", gen)
                self.logger.info("Invalid code:\n %s", child_program, stacklevel=2)
                continue

            # Write to temp file for evaluator
            tmp_path = Path(".alpha_tmp.py")
            tmp_path.write_text(child_program)
            score = self.problem.evaluate(str(tmp_path))
            pid = self.database.add(child_program, score, gen, parent_row["id"])
            self.logger.info("Gen %d: new score %.3f (id %d)", gen, score, pid)
