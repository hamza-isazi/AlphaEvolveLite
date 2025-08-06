
from pathlib import Path
import concurrent.futures
from typing import List
from tqdm import tqdm
import statistics
import logging
import traceback
from .log import init_logger
from .config import Config
from .program_generator import generate_program
from .db import EvolutionaryDatabase, ProgramRecord
from .llm import LLMEngine, create_llm_client
from .problem import Problem
from .prompts import PromptSampler
from .patcher import PatchApplier

class ControllerContext:
    """Context object that provides centralized access to configuration and common dependencies."""
    
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger        
        self.database = EvolutionaryDatabase(self.cfg)
        self.problem = Problem(self.cfg.problem_entry, self.cfg.problem_eval)
        self.prompt_sampler = PromptSampler(self.database, enable_feedback=self.cfg.evolution.enable_feedback)
        self.patcher = PatchApplier()
        self.client = create_llm_client(self.cfg.llm)

class EvolutionController:
    """
    Controller for running the evolutionary algorithm.
    """
    
    def __init__(self, cfg: Config, resume: bool = False):
        self.cfg = cfg
        self.resume = resume
        self.logger = init_logger(debug=cfg.debug)
        self.logger.info("Connected to database at %s", cfg.db_uri)
        self.context = ControllerContext(cfg, self.logger)

        if not resume:
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
            self.current_gen = 1 # Gen 0 is the seed program, we start at 1
        else:
            # Get current generation from database
            with self.context.database.get_connection() as conn:
                self.current_gen = self.context.database.get_latest_generation(conn) + 1
            self.logger.info("Resuming from generation %d", self.current_gen)

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


    def run_evolution(self):
        """
        Execute the main evolutionary algorithm to generate and evolve programs.
        
        This function implements a continuous evolution strategy where:
        - Programs are generated concurrently using a thread pool
        - Each generation produces a population of programs
        - The best programs from each generation are tracked
        - Evolution continues until max_generations * population_size programs are created
        
        The algorithm maintains a continuous flow of program generation and evaluation,
        with progress tracking and logging throughout the process.
        """
        # Initialize evolution parameters
        max_concurrent = self.cfg.evolution.population_size  # Number of concurrent program generations
        max_total = self.cfg.evolution.max_generations * self.cfg.evolution.population_size  # Total programs to generate
        completed = 0  # Counter for completed program generations
        
        # Set number of completed programs based on resume flag
        if self.resume:
            completed = self.current_gen * self.cfg.evolution.population_size  # Account for already completed programs
            self.logger.info("Resuming evolution from generation %d (completed: %d programs)", self.current_gen, completed)
        else:
            completed = 0  # Counter for completed program generations
            
        generation_program_records = []  # Store results for current generation
        
        # Get the current best score from the database as baseline
        with self.context.database.get_connection() as conn:
            best_score = self.context.database.top_k(conn, 1)[0]["score"]
        task_id_counter = completed  # Unique identifier for each generation task

        self.logger.info("Starting continuous evolution with up to %d concurrent individuals", max_concurrent)
        gen_pbar = tqdm(total=self.cfg.evolution.population_size, desc=f"Generation {self.current_gen}")

        # Helper function to generate a single program and save it to the database
        def generate_and_save_program(task_id):
            """
            Generate a single program using evolutionary techniques and store it in the database.
            
            Args:
                task_id: Unique identifier for this generation task
                
            Returns:
                ProgramRecord or None: The generated program record if successful, None if failed
            """
            try:
                # Sample parent program and inspiration programs from database
                parent_row, inspiration_rows = self.context.database.sample()
                
                # Generate a new program using the evolutionary algorithm
                result = generate_program(
                    task_id,
                    (parent_row, inspiration_rows),
                    self.current_gen,
                    self.cfg,
                    self.logger,
                    self.context.client
                )
                
                # Store the generated program in the database
                program_id = self.context.database.add(result)
                
                # Log the result based on whether generation was successful
                if result.score is not None:
                    self.logger.debug("Stored successful program (program_id=%d, task_id=%d, model=%s) with score %.3f", program_id, task_id, result.used_model, result.score)
                else:
                    self.logger.debug("Stored failed program (program_id=%d, task_id=%d, model=%s) with failure type (%s) and error: %s", program_id, task_id, result.used_model, result.failure_type, result.error_message)
                return result
            except Exception as e:
                # Log any exceptions that occur during program generation
                self.logger.error("Individual %d failed: %s", task_id, str(e))
                self.logger.error("Full traceback: %s", traceback.format_exc())
                return None

        # Set up thread pool for concurrent program generation
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit initial batch of program generation tasks
            futures = {
                executor.submit(generate_and_save_program, task_id_counter + i)
                for i in range(max_concurrent)
            }
            task_id_counter += max_concurrent

            # Main evolution loop: generate programs continuously until max_total is reached
            while completed < max_total:
                # Wait for at least one task to complete before proceeding
                # `futures` is automatically updated to only include still-running tasks
                done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

                # Process completed program generation tasks
                for finished in done:
                    result = finished.result()
                    # Store program if successful
                    if result:
                        generation_program_records.append(result)
                    # Update progress
                    completed += 1
                    gen_pbar.update(1)

                    # Check if we've completed a full generation
                    if completed % self.cfg.evolution.population_size == 0:
                        # Close current progress bar and log generation summary
                        gen_pbar.close()
                        self._log_generation_summary(
                            self.current_gen,
                            self.cfg.evolution.population_size,
                            generation_program_records
                        )
                        
                        # Move to next generation
                        self.current_gen += 1
                        
                        # Check if this generation produced a new best score
                        scores = [r.score for r in generation_program_records if r.score is not None]
                        gen_best_score = max(scores) if scores else None
                        if gen_best_score is not None and gen_best_score > best_score:
                            self.logger.info("New best score: %.3f (prev: %.3f)", gen_best_score, best_score)
                            best_score = gen_best_score
                        
                        # Reset for next generation
                        generation_program_records.clear()
                        
                        # Create new progress bar for next generation (if not the last one)
                        if completed < max_total - 1:
                            gen_pbar = tqdm(total=self.cfg.evolution.population_size, desc=f"Generation {self.current_gen}")

                    # Submit a new task to replace the completed one, maintaining continuous program generation
                    future = executor.submit(generate_and_save_program, task_id_counter)
                    futures.add(future)
                    task_id_counter += 1

        # Log final evolution results
        self.logger.info("Continuous evolution finished after %d individuals. Best score: %.3f", completed, best_score)
        
        # Save the top performing candidates to files
        self.save_top_k_candidates()

    def save_top_k_candidates(self):
        self.logger.info("Saving top %d candidates", self.cfg.exp.save_top_k)
        with self.context.database.get_connection() as conn:
            rows = self.context.database.top_k(conn, self.cfg.exp.save_top_k)
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
