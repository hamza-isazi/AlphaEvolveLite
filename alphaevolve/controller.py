
from pathlib import Path
import tempfile
import traceback

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

    def _apply_patch_with_retries(self, parent_row: dict, initial_diff: str) -> tuple[str | None, int]:
        """
        Apply a patch with retry logic for patch application failures.
        
        Returns:
            Tuple of (child_program, retry_count)
        """
        retry_count = 0
        child_program = None
        diff = initial_diff
        
        while retry_count <= self.cfg.evolution.max_patch_retries:
            if retry_count > 0:
                # This is a retry - generate a retry prompt
                retry_prompt = self.prompt_sampler.build_patch_retry_prompt(parent_row)
                diff = self.llm.generate(retry_prompt)
            
            child_program = self.patcher.apply_diff(parent_row["code"], diff)
            
            if child_program and self.patcher.is_valid(child_program):
                break
            
            if retry_count < self.cfg.evolution.max_patch_retries:
                retry_count += 1
                self.logger.info("Gen %d: patch failed, retrying (%d/%d)", 
                               self.current_gen, retry_count, self.cfg.evolution.max_patch_retries)
            else:
                self.logger.info("Gen %d: patch failed, ran out of retries", 
                               self.current_gen)
        
        return child_program, retry_count

    def _evaluate_with_retries(self, parent_row: dict, child_program: str) -> tuple[float | None, str | None, int]:
        """
        Evaluate a program with retry logic for evaluation failures.
        
        Returns:
            Tuple of (score, final_child_program, retry_count)
        """
        retry_count = 0
        score = None
        error_message = "Unknown error"
        current_program = child_program
        
        while retry_count <= self.cfg.evolution.max_eval_retries:
            if retry_count > 0:
                # This is a retry - generate a retry prompt for evaluation error
                retry_prompt = self.prompt_sampler.build_eval_retry_prompt(current_program, error_message)
                diff = self.llm.generate(retry_prompt)
                
                # Apply the new diff
                new_child_program = self.patcher.apply_diff(current_program, diff)
                if new_child_program and self.patcher.is_valid(new_child_program):
                    current_program = new_child_program
                else:
                    # If the retry diff fails, break out of retry loop
                    break
            
            try:
                # Write to temp file for evaluator
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as tmp:
                    tmp.write(current_program)
                    tmp.flush()
                    score = self.problem.evaluate(tmp.name)
                break  # Success - exit retry loop
            except Exception as e:
                # Get traceback and filter to only include frames from within evaluate function
                tb = traceback.extract_tb(e.__traceback__)
                relevant_frames = []
                in_evaluate = False
                
                for frame in tb:
                    # Check if we're entering the evaluate function
                    if frame.name == 'evaluate' and ('problem.py' in frame.filename or 'evaluate.py' in frame.filename):
                        in_evaluate = True
                    
                    # Only include frames once we're inside evaluate function
                    if in_evaluate:
                        relevant_frames.append(frame)
                
                # Format the traceback
                tb_lines = []
                for frame in relevant_frames:
                    tb_lines.append(f"  File \"{frame.filename}\", line {frame.lineno}, in {frame.name}")
                    if frame.line:
                        tb_lines.append(f"    {frame.line.strip()}")
                
                error_message = f"{str(e)}\nTraceback:\n" + "\n".join(tb_lines)
                
                if retry_count < self.cfg.evolution.max_eval_retries:
                    retry_count += 1
                    self.logger.info("Gen %d: evaluation failed, retrying (%d/%d): %s", 
                                   self.current_gen, retry_count, self.cfg.evolution.max_eval_retries, error_message)
                else:
                    self.logger.info("Gen %d: evaluation failed, ran out of retries: %s", 
                        self.current_gen, error_message)
        
        return score, current_program, retry_count

    def _run_single_generation(self, gen: int) -> bool:
        """
        Run a single generation with retry logic.
        
        Returns:
            True if generation was successful, False otherwise
        """
        self.current_gen = gen
        parent_row, inspiration_rows = self.database.sample()
        
        # Reset conversation for new generation
        self.llm.reset_conversation()
        
        # Initial prompt
        prompt = self.prompt_sampler.build(parent_row, inspiration_rows)
        initial_diff = self.llm.generate(prompt)
        
        # Apply patch with retries
        child_program, patch_retries = self._apply_patch_with_retries(parent_row, initial_diff)
        
        if not child_program or not self.patcher.is_valid(child_program):
            self.logger.info("Gen %d: invalid patch after %d retries, skipping", gen, patch_retries)
            return False

        # Evaluate with retries
        score, final_program, eval_retries = self._evaluate_with_retries(parent_row, child_program)
        
        if score is None:
            self.logger.info("Gen %d: evaluation failed after %d retries, skipping", gen, eval_retries)
            return False

        # Success - add to database
        assert final_program is not None  # Should be guaranteed by success check above
        pid = self.database.add(final_program, score, gen, parent_row["id"])
        self.logger.info("Gen %d: new score %.3f (id %d)", gen, score, pid)
        return True

    def run_evolution(self):
        self.current_gen = 0
        for gen in range(1, self.cfg.evolution.max_generations + 1):
            success = self._run_single_generation(gen)
            if not success:
                self.logger.info("Gen %d: generation failed, continuing to next generation", gen)
        
        self.logger.info("Evolution complete. Final generation: %d", self.cfg.evolution.max_generations)
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
