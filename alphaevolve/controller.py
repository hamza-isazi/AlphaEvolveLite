
from pathlib import Path
import concurrent.futures
from typing import List, Dict, Any
from tqdm import tqdm
import statistics
from .log import init_logger, get_logger
from .patcher import PatchApplier
from .prompts import PromptSampler
from .problem import Problem
from .db import EvolutionaryDatabase
from .config import Config
from .llm import OpenAIEngine
from .program_generator import generate_program


class EvolutionController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = init_logger(debug=cfg.debug)
        self.database = EvolutionaryDatabase(cfg)
        self.logger.info("Connected to database at %s", cfg.db_uri)
        self.problem = Problem(cfg.problem_entry, cfg.problem_eval)
        self.prompt_sampler = PromptSampler(self.database)
        self.llm = OpenAIEngine(cfg.llm)
        self.patcher = PatchApplier()

        # Seed archive with original solution
        seed_code = Path(cfg.problem_entry).read_text()
        seed_score = self.problem.evaluate_with_timeout(cfg.problem_entry, cfg.evolution.evaluation_timeout)
        if seed_score is None:
            self.logger.error("Seed evaluation timed out after %.1f seconds", cfg.evolution.evaluation_timeout)
            raise RuntimeError("Seed evaluation timed out")
        self.database.add(code=seed_code, explanation="Initial seed program", score=seed_score, gen=0, parent_id=None)
        self.logger.info("Seed score %.3f", seed_score)

    def _calculate_generation_stats(self, generation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate summary statistics for a generation.
        
        Parameters
        ----------
        generation_results : List[Dict[str, Any]]
            List of individual results from the generation
            
        Returns
        -------
        Dict[str, Any]
            Summary statistics including fitness metrics and error counts
        """
        if not generation_results:
            return {
                "total_individuals": 0,
                "successful_individuals": 0,
                "success_rate": 0.0,
                "avg_fitness": 0.0,
                "best_fitness": 0.0,
                "evaluation_failures": 0,
                "timeouts": 0,
                "patch_failures": 0,
                "invalid_response_failures": 0
            }
        
        successful_results = [r for r in generation_results if r["score"] is not None]
        failed_results = [r for r in generation_results if r["score"] is None]
        
        # Calculate fitness statistics
        fitness_scores = [r["score"] for r in successful_results]
        avg_fitness = statistics.mean(fitness_scores) if fitness_scores else 0.0
        best_fitness = max(fitness_scores) if fitness_scores else 0.0
        
        # Count different types of failures
        syntax_errors = sum(1 for r in failed_results if r["failure_type"] == "syntax_error")
        evaluation_failures = sum(1 for r in failed_results if r["failure_type"] == "runtime_error")
        timeouts = sum(1 for r in failed_results if r["failure_type"] == "timeout")
        patch_failures = sum(1 for r in failed_results if r["failure_type"] == "patch_failure")
        invalid_response_failures = sum(1 for r in failed_results if r["failure_type"] == "invalid_response")

        return {
            "total_individuals": len(generation_results),
            "successful_individuals": len(successful_results),
            "success_rate": len(successful_results) / len(generation_results) * 100,
            "avg_fitness": avg_fitness,
            "best_fitness": best_fitness,
            "syntax_errors": syntax_errors,
            "evaluation_failures": evaluation_failures,
            "timeouts": timeouts,
            "patch_failures": patch_failures,
            "invalid_response_failures": invalid_response_failures
        }

    def _log_generation_summary(self, gen: int, stats: Dict[str, Any]) -> None:
        """
        Log summary statistics for a generation.
        
        Parameters
        ----------
        gen : int
            Generation number
        stats : Dict[str, Any]
            Generation statistics
        """
        self.logger.info("=" * 60)
        self.logger.info("Generation %d Summary:", gen)
        self.logger.info("  Success Rate: %d/%d (%.1f%%)", 
                        stats["successful_individuals"], stats["total_individuals"], stats["success_rate"])
        self.logger.info("  Fitness - Avg: %.3f, Best: %.3f", stats["avg_fitness"], stats["best_fitness"])
        self.logger.info("  Failures - Syntax Errors: %d (%.1f%%), Evaluation: %d (%.1f%%), Timeouts: %d (%.1f%%), Patches: %d (%.1f%%), Invalid Responses: %d (%.1f%%)",
                        stats["syntax_errors"], stats["syntax_errors"]/stats["total_individuals"]*100,
                        stats["evaluation_failures"], stats["evaluation_failures"]/stats["total_individuals"]*100,
                        stats["timeouts"], stats["timeouts"]/stats["total_individuals"]*100,
                        stats["patch_failures"], stats["patch_failures"]/stats["total_individuals"]*100,
                        stats["invalid_response_failures"], stats["invalid_response_failures"]/stats["total_individuals"]*100)
        self.logger.info("=" * 60)

    def _run_single_generation(self) -> int:
        """
        Run a single generation, generating multiple individuals in parallel.
        
        Returns:
            Number of successful individuals generated
        """
        population_size = self.cfg.evolution.population_size
        
        self.logger.debug("Gen %d: Starting generation with population size %d", self.current_gen, population_size)
        
        # Pre-sample all parents and inspirations to avoid database threading issues
        parent_inspiration_pairs = []
        for i in range(population_size):
            try:
                parent_row, inspiration_rows = self.database.sample()
                parent_inspiration_pairs.append((parent_row, inspiration_rows))
            except Exception as e:
                # This is a fundamental error - no parents available means evolution cannot continue
                raise RuntimeError(f"Gen {self.current_gen}: No parents available for evolution. Database may be empty or corrupted.")
        
        # Generate programs in parallel with progress bar
        generation_results = []
        successful_individuals = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(population_size, 80)) as executor:
            # Submit all program generation tasks with pre-sampled data
            future_to_id = {
                executor.submit(generate_program, i, parent_data, self.current_gen, self.cfg, get_logger()): i 
                for i, parent_data in enumerate(parent_inspiration_pairs)
            }
            
            # Use tqdm for progress tracking
            with tqdm(total=population_size, desc=f"Generation {self.current_gen}", 
                     disable=self.cfg.debug) as pbar:
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_id):
                    individual_id = future_to_id[future]
                    try:
                        result = future.result()
                        
                        # Add program to database (successful or failed)
                        pid = self.database.add(
                            result["program"],
                            result["explanation"],
                            result["score"], 
                            self.current_gen, 
                            result["parent_id"],
                            result["failure_type"]
                        )
                        
                        if result["score"] is not None:
                            successful_individuals += 1
                        
                        generation_results.append({
                            "score": result["score"],
                            "failure_type": result["failure_type"],
                            "individual_id": individual_id,
                            "db_id": pid
                        })
                    except Exception as e:
                        self.logger.error("Gen %d, Individual %d: failed with exception: %s", 
                                        self.current_gen, individual_id, str(e))
                    
                    pbar.update(1)
        
        # Calculate and log generation statistics
        stats = self._calculate_generation_stats(generation_results)
        self._log_generation_summary(self.current_gen, stats)
        
        self.logger.debug("Gen %d: Completed with %d/%d successful individuals", 
                        self.current_gen, successful_individuals, population_size)
        return successful_individuals

    def run_evolution(self):
        self.current_gen = 0
        total_successful = 0
        
        self.logger.info("Starting evolution with %d generations, population size %d", 
                        self.cfg.evolution.max_generations, self.cfg.evolution.population_size)
        
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
