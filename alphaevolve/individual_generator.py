import tempfile
import traceback
import logging
from concurrent.futures import TimeoutError
from typing import List, Tuple, Optional, Dict, Any

from .patcher import PatchApplier
from .prompts import PromptSampler
from .problem import Problem
from .llm import OpenAIEngine
from .config import Config


def apply_patch_with_retries(
    parent_row: dict, 
    initial_diff: str, 
    llm_instance: OpenAIEngine, 
    individual_id: int,
    current_gen: int,
    max_patch_retries: int,
    prompt_sampler: PromptSampler,
    patcher: PatchApplier,
    logger: logging.Logger
) -> str | None:
    """
    Apply a patch with retry logic for patch application failures.
    
    Returns:
        str | None: child_program if successful, None if failed
    """
    retry_count = 0
    child_program = None
    diff = initial_diff
    
    while retry_count <= max_patch_retries:
        if retry_count > 0:
            # This is a retry - generate a retry prompt
            retry_prompt = prompt_sampler.build_patch_retry_prompt(parent_row)
            diff = llm_instance.generate(retry_prompt)
        
        child_program = patcher.apply_diff(parent_row["code"], diff)
        
        if child_program and patcher.is_valid(child_program):
            break
        
        retry_count += 1
        if retry_count <= max_patch_retries:
            logger.info("Gen %d, Individual %d: patch failed, retrying (%d/%d)", 
                       current_gen, individual_id, retry_count, max_patch_retries)
        else:
            logger.info("Gen %d, Individual %d: patch failed, ran out of retries", 
                       current_gen, individual_id)
    
    return child_program


def evaluate_with_retries(
    child_program: str, 
    llm_instance: OpenAIEngine, 
    individual_id: int,
    current_gen: int,
    max_eval_retries: int,
    prompt_sampler: PromptSampler,
    patcher: PatchApplier,
    problem: Problem,
    evaluation_timeout: float,
    logger: logging.Logger
) -> tuple[float | None, str | None]:
    """
    Evaluate a program with retry logic for evaluation failures.
    
    Returns:
        Tuple of (score, final_child_program)
    """
    retry_count = 0
    score = None
    error_message = "Unknown error"
    current_program = child_program
    
    while retry_count <= max_eval_retries:
        if retry_count > 0:
            # This is a retry - generate a retry prompt for evaluation error
            retry_prompt = prompt_sampler.build_eval_retry_prompt(current_program, error_message)
            diff = llm_instance.generate(retry_prompt)
            
            # Apply the new diff
            new_child_program = patcher.apply_diff(current_program, diff)
            if new_child_program and patcher.is_valid(new_child_program):
                current_program = new_child_program
            else:
                # If the retry diff fails, break out of retry loop
                break
        
        try:
            # Write to temp file for evaluator
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as tmp:
                tmp.write(current_program)
                tmp.flush()
                
                # Run evaluation with timeout
                score = problem.evaluate_with_timeout(tmp.name, evaluation_timeout)
                
            break  # Success - exit retry loop
        except TimeoutError:
            error_message = f"Evaluation timed out after {evaluation_timeout} seconds"
        except Exception as e:
            error_message = str(e)
        retry_count += 1
        if retry_count <= max_eval_retries:
            logger.info("Gen %d, Individual %d: evaluation failed, retrying (%d/%d): %s", 
                        current_gen, individual_id, retry_count, max_eval_retries, error_message)
        else:
            logger.info("Gen %d, Individual %d: evaluation failed, ran out of retries: %s", 
                        current_gen, individual_id, error_message)
    
    return score, current_program


def generate_single_individual(
    individual_id: int, 
    parent_data: Tuple[dict, List[dict]], 
    current_gen: int,
    cfg: Config,
    logger: logging.Logger
) -> Optional[Tuple[float, str, int]]:
    """
    Generate a single individual for the population using pre-sampled data.
    
    This function is separated from the main controller to avoid SQLite connection 
    pickling issues when using ProcessPoolExecutor. It contains all the logic needed
    to generate and evaluate a single individual in isolation.
    
    Returns:
        Tuple of (score, program, parent_id) if successful, None if failed
    """
    try:
        parent_row, inspiration_rows = parent_data
        
        if parent_row is None:
            logger.info("Gen %d, Individual %d: no parent data available, skipping", current_gen, individual_id)
            return None
        
        # Create instances for this individual to avoid conflicts
        individual_llm = OpenAIEngine(cfg.llm)
        patcher = PatchApplier()
        problem = Problem(cfg.problem_entry, cfg.problem_eval)
        prompt_sampler = PromptSampler(None)  # No database needed for prompt building
        
        # Initial prompt
        prompt = prompt_sampler.build(parent_row, inspiration_rows)
        initial_diff = individual_llm.generate(prompt)
        
        # Apply patch with retries
        child_program = apply_patch_with_retries(
            parent_row, initial_diff, individual_llm, individual_id, 
            current_gen, cfg.evolution.max_patch_retries, prompt_sampler, patcher, logger
        )
        
        if not child_program or not patcher.is_valid(child_program):
            logger.info("Gen %d, Individual %d: invalid patch, skipping", current_gen, individual_id)
            return None

        # Evaluate with retries
        score, final_program = evaluate_with_retries(
            child_program, individual_llm, individual_id, current_gen,
            cfg.evolution.max_eval_retries, prompt_sampler, patcher, problem, 
            cfg.evolution.evaluation_timeout, logger
        )
        
        if score is None or final_program is None:
            logger.info("Gen %d, Individual %d: evaluation failed, skipping", current_gen, individual_id)
            return None

        # Success
        logger.info("Gen %d, Individual %d: new score %.3f", current_gen, individual_id, score)
        return score, final_program, parent_row["id"]
        
    except Exception as e:
        logger.error("Gen %d, Individual %d: unexpected error: %s", current_gen, individual_id, str(e))
        return None 