
from pathlib import Path
import concurrent.futures
from typing import List
from tqdm import tqdm
import statistics
from .log import init_logger
from .config import Config, ConfigContext
from .program_generator import generate_program
from .db import ProgramRecord
from .llm import LLMEngine


class EvolutionController:
    """
    Controller for running the evolutionary algorithm.
    """
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.context = ConfigContext(cfg)
        self.logger = init_logger(debug=cfg.debug)
        self.logger.info("Connected to database at %s", cfg.db_uri)
        
        # Seed archive with original solution
        seed_code = Path(cfg.problem_entry).read_text()
        seed_score, seed_execution_time, seed_logs = self.context.problem.evaluate_with_timeout(cfg.problem_entry, cfg.evolution.eval_timeout) 
        seed_record = ProgramRecord(
            gen=0,
            parent_id=None,
            code=seed_code, 
            explanation="Initial seed program", 
            score=seed_score,
            total_evaluation_time=seed_execution_time,
            evaluation_logs=seed_logs
        )
        # Generate feedback for the seed program if enabled
        if cfg.evolution.enable_feedback:
            feedback_prompt = self.context.prompt_sampler.build_feedback_prompt(seed_record.code, seed_record.score, seed_record.evaluation_logs, cfg.problem_eval)
            llm_instance = LLMEngine(cfg.llm, self.context.client, self.logger)
            seed_record.used_model = llm_instance.get_used_model()
            seed_record.feedback = llm_instance.generate(feedback_prompt)
            seed_record.total_llm_time, seed_record.total_tokens = llm_instance.get_metrics()
        self.context.database.add(seed_record)
        self.logger.info("Seed score %.3f", seed_score)

    def _log_generation_summary(self, gen: int, population_size: int, program_records: List[ProgramRecord]) -> None:
        """Log summary statistics for a generation."""
        if not program_records:
            self.logger.info("=" * 60)
            self.logger.info("Generation %d Summary: No programs generated", gen)
            self.logger.info("=" * 60)
            return
        
        successful_records = [r for r in program_records if r.score is not None]
        failed_records = [r for r in program_records if r.score is None]
        
        # Calculate fitness statistics
        fitness_scores = [r.score for r in successful_records]
        avg_fitness = statistics.mean(fitness_scores) if fitness_scores else 0.0
        best_fitness = max(fitness_scores) if fitness_scores else 0.0
        
        # Calculate retry and total evaluation time statistics
        retry_counts = [r.retry_count for r in program_records]
        avg_retries = statistics.mean(retry_counts) if retry_counts else 0.0
        
        total_evaluation_times = [r.total_evaluation_time for r in program_records if r.total_evaluation_time is not None]
        avg_total_evaluation_time = statistics.mean(total_evaluation_times) if total_evaluation_times else 0.0
        
        # Calculate generation time and total LLM time statistics
        generation_times = [r.generation_time for r in program_records if r.generation_time is not None]
        avg_generation_time = statistics.mean(generation_times) if generation_times else 0.0
        
        total_llm_times = [r.total_llm_time for r in program_records if r.total_llm_time is not None]
        avg_total_llm_time = statistics.mean(total_llm_times) if total_llm_times else 0.0
        
        # Calculate total token statistics
        total_tokens_list = [r.total_tokens for r in program_records if r.total_tokens is not None]
        avg_total_tokens = statistics.mean(total_tokens_list) if total_tokens_list else 0.0
        
        # Count different types of failures
        syntax_errors = sum(1 for r in failed_records if r.failure_type == "syntax_error")
        evaluation_failures = sum(1 for r in failed_records if r.failure_type == "runtime_error")
        timeouts = sum(1 for r in failed_records if r.failure_type == "timeout")
        patch_failures = sum(1 for r in failed_records if r.failure_type == "patch_failure")
        invalid_response_failures = sum(1 for r in failed_records if r.failure_type == "invalid_response")
        
        successful_individuals = len(successful_records)
        success_rate = successful_individuals / population_size * 100 if population_size > 0 else 0.0
        
        self.logger.info("=" * 60)
        self.logger.info("Generation %d Summary:", gen)
        self.logger.info("  Success Rate: %d/%d (%.1f%%)", 
                        successful_individuals, population_size, success_rate)
        self.logger.info("  Fitness - Avg: %.3f, Best: %.3f", avg_fitness, best_fitness)
        self.logger.info("  Performance - Avg Retries: %.1f, Avg Total Eval Time: %.2fs, Avg Gen Time: %.2fs, Avg Total LLM Time: %.2fs, Avg Total Tokens: %.0f", 
                        avg_retries, avg_total_evaluation_time, avg_generation_time, avg_total_llm_time, avg_total_tokens)
        self.logger.info("  Failures - Syntax Errors: %d (%.1f%%), Evaluation: %d (%.1f%%), Timeouts: %d (%.1f%%), Patches: %d (%.1f%%), Invalid Responses: %d (%.1f%%)",
                        syntax_errors, syntax_errors/population_size*100,
                        evaluation_failures, evaluation_failures/population_size*100,
                        timeouts, timeouts/population_size*100,
                        patch_failures, patch_failures/population_size*100,
                        invalid_response_failures, invalid_response_failures/population_size*100)
        self.logger.info("=" * 60)

    def run_generation(self, current_gen: int) -> int:
        """
        Run a single generation, generating multiple individuals in parallel.
        
        Returns:
            Number of successful individuals generated
        """
        population_size = self.cfg.evolution.population_size
        
        self.logger.debug("Gen %d: Starting generation with population size %d", current_gen, population_size)
        
        # Pre-sample all parents and inspirations to avoid database threading issues
        parent_inspiration_pairs = []
        for i in range(population_size):
            try:
                parent_row, inspiration_rows = self.context.database.sample()
                parent_inspiration_pairs.append((parent_row, inspiration_rows))
            except Exception as e:
                # This is a fundamental error - no parents available means evolution cannot continue
                raise RuntimeError(f"Gen {current_gen}: No parents available for evolution. Database may be empty or corrupted.")
        
        # Generate programs in parallel with progress bar
        program_records = []
        successful_individuals = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(population_size, 20)) as executor:
            # Submit all program generation tasks with pre-sampled data
            # Each worker will create its own program generation context to avoid sharing conversation history
            future_to_id = {
                executor.submit(generate_program, i, parent_data, current_gen, self.cfg, self.logger, self.context.client): i 
                for i, parent_data in enumerate(parent_inspiration_pairs)
            }
            
            # Use tqdm for progress tracking
            with tqdm(total=population_size, desc=f"Generation {current_gen}", 
                     disable=self.cfg.debug) as pbar:
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_id):
                    individual_id = future_to_id[future]
                    try:
                        result = future.result()
                        
                        # Add program to database
                        self.context.database.add(result)
                        
                        # Store successful individuals and program records for logging
                        if result.score is not None:
                            successful_individuals += 1                        
                        program_records.append(result)
                    except Exception as e:
                        import traceback
                        self.logger.error("Gen %d, Individual %d: failed with exception: %s", 
                                        current_gen, individual_id, str(e))
                        self.logger.error("Full traceback: %s", traceback.format_exc())
                    
                    pbar.update(1)
        
        # Log generation statistics
        self._log_generation_summary(current_gen, population_size, program_records)
        
        self.logger.debug("Gen %d: Completed with %d/%d successful individuals", 
                        current_gen, successful_individuals, population_size)
        return successful_individuals

    def run_evolution(self):
        total_successful = 0
        
        self.logger.info("Starting evolution with %d generations, population size %d", 
                        self.cfg.evolution.max_generations, self.cfg.evolution.population_size)
        
        for gen in range(1, self.cfg.evolution.max_generations + 1):
            successful = self.run_generation(gen)
            total_successful += successful
            
            if successful == 0:
                self.logger.warning("Gen %d: no successful individuals generated, continuing to next generation", gen)
        
        self.logger.info("Evolution complete. Final generation: %d, Total successful individuals: %d", 
                        self.cfg.evolution.max_generations, total_successful)
        self.save_top_k_candidates()

    def save_top_k_candidates(self):
        self.logger.info("Saving top %d candidates", self.cfg.exp.save_top_k)
        rows = self.context.database.top_k(self.cfg.exp.save_top_k)
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
