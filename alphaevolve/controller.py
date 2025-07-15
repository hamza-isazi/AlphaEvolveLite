
from pathlib import Path
import concurrent.futures
from typing import List, Tuple, Optional

from .log import init_logger
from .patcher import PatchApplier
from .prompts import PromptSampler
from .problem import Problem
from .db import EvolutionaryDatabase
from .config import Config
from .llm import OpenAIEngine
from .individual_generator import generate_single_individual


class EvolutionController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = init_logger()
        self.database = EvolutionaryDatabase(cfg)
        self.logger.info("Connected to database at %s", cfg.db_uri)
        self.problem = Problem(cfg.problem_entry, cfg.problem_eval)
        self.prompt_sampler = PromptSampler(self.database)
        self.llm = OpenAIEngine(cfg.llm)
        self.patcher = PatchApplier()

        # Seed archive with original solution
        seed_code = Path(cfg.problem_entry).read_text()
        seed_score = self.problem.evaluate(cfg.problem_entry)
        self.database.add(seed_code, seed_score, gen=0, parent_id=None)
        self.logger.info("Seed score %.3f", seed_score)



    def _run_single_generation(self) -> int:
        """
        Run a single generation, generating multiple individuals in parallel.
        
        Returns:
            Number of successful individuals generated
        """
        population_size = self.cfg.evolution.population_size
        
        self.logger.info("Gen %d: Starting generation with population size %d", self.current_gen, population_size)
        
        # Pre-sample all parents and inspirations to avoid database threading issues
        parent_inspiration_pairs = []
        for i in range(population_size):
            try:
                parent_row, inspiration_rows = self.database.sample()
                parent_inspiration_pairs.append((parent_row, inspiration_rows))
            except Exception as e:
                self.logger.error("Gen %d: failed to sample parent/inspiration for individual %d: %s", 
                                self.current_gen, i, str(e))
                # Use empty data as fallback
                parent_inspiration_pairs.append((None, []))
        
        # Generate individuals in parallel
        successful_individuals = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(population_size, 4)) as executor:
            # Submit all individual generation tasks with pre-sampled data
            future_to_id = {
                executor.submit(generate_single_individual, i, parent_data, self.current_gen, self.cfg, self.logger): i 
                for i, parent_data in enumerate(parent_inspiration_pairs)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_id):
                individual_id = future_to_id[future]
                try:
                    result = future.result()
                    if result is not None:
                        score, program, parent_id = result
                        # Add to database
                        pid = self.database.add(program, score, self.current_gen, parent_id)
                        successful_individuals += 1
                        self.logger.info("Gen %d, Individual %d: added to database with id %d", 
                                        self.current_gen, individual_id, pid)
                except Exception as e:
                    self.logger.error("Gen %d, Individual %d: failed with exception: %s", 
                                    self.current_gen, individual_id, str(e))
        
        self.logger.info("Gen %d: Completed with %d/%d successful individuals", 
                        self.current_gen, successful_individuals, population_size)
        return successful_individuals

    def run_evolution(self):
        self.current_gen = 0
        total_successful = 0
        
        for gen in range(1, self.cfg.evolution.max_generations + 1):
            self.current_gen = gen
            successful = self._run_single_generation()
            total_successful += successful
            
            if successful == 0:
                self.logger.warning("Gen %d: no successful individuals generated, continuing to next generation", self.current_gen)
        
        self.logger.info("Evolution complete. Final generation: %d, Total successful individuals: %d", 
                        self.cfg.evolution.max_generations, total_successful)
        self.save_top_k_candidates()

    def save_top_k_candidates(self):
        self.logger.info("Saving top %d candidates", self.cfg.exp.save_top_k)
        rows = self.database.top_k(self.cfg.exp.save_top_k)
        results_dir = Path(f"results/{self.cfg.exp.label}")
        results_dir.mkdir(parents=True, exist_ok=True)

        for row in rows:
            fname = (
                f"{row['experiment_id']}_gen{row['gen']}_"
                f"score{row['score']:.3f}_id{row['id']}.py"
            )
            out_path = results_dir / fname
            out_path.write_text(row["code"])
            self.logger.info("Saved top candidate to %s", out_path)
