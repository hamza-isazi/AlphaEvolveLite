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
from .response_parser import parse_structured_response


def apply_patch_with_retries(
    parent_row: dict, 
    initial_code: str, 
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
    code_section = initial_code
    
    while retry_count <= max_patch_retries:
        if retry_count > 0:
            # This is a retry - generate a retry prompt
            retry_prompt = prompt_sampler.build_patch_retry_prompt(parent_row)
            response = llm_instance.generate(retry_prompt)
            
            # For retries, just extract code (no explanation needed)
            _, code_section = parse_structured_response(response)
            if code_section is None:
                # If no code found, break out of retry loop
                break
        
        child_program = patcher.apply_diff(parent_row["code"], code_section)
        
        if child_program:
            break
        
        retry_count += 1
        if retry_count <= max_patch_retries:
            logger.debug("Gen %d, Individual %d: patch failed, retrying (%d/%d)", 
                       current_gen, individual_id, retry_count, max_patch_retries)
        else:
            logger.debug("Gen %d, Individual %d: patch failed, ran out of retries", 
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
) -> tuple[float | None, str, str | None]:
    """
    Evaluate a program with retry logic for evaluation failures.
    
    Returns:
        Tuple of (score, final_child_program, failure_type)
        failure_type is None if successful, otherwise describes the failure
    """
    retry_count = 0
    score = None
    error_message = "Unknown error"
    current_program = child_program
    failure_type = None
    
    while retry_count <= max_eval_retries:
        if retry_count > 0:
            # This is a retry - generate a retry prompt for evaluation error
            retry_prompt = prompt_sampler.build_eval_retry_prompt(current_program, error_message)
            response = llm_instance.generate(retry_prompt)
            
            # For retries, just extract code (no explanation needed)
            _, code_section = parse_structured_response(response)
            if code_section is None:
                # If no code found, break out of retry loop
                failure_type = "invalid_response"
                break
            
            # Apply the new diff
            new_child_program = patcher.apply_diff(current_program, code_section)
            if new_child_program:
                current_program = new_child_program
            else:
                # If the retry diff fails, break out of retry loop
                failure_type = "secondary_patch_failure"
                break
        
        try:
            # Compile the program to check for syntax errors
            compile(current_program, "<candidate>", "exec")
            # Write to temp file for evaluator
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as tmp:
                tmp.write(current_program)
                tmp.flush()
                
                # Run evaluation with timeout
                score = problem.evaluate_with_timeout(tmp.name, evaluation_timeout)
                
            break  # Success - exit retry loop
        except SyntaxError as e:
            error_message = f"Syntax error: {str(e)}"
            failure_type = "syntax_error"
        except TimeoutError:
            error_message = f"Evaluation timed out after {evaluation_timeout} seconds"
            failure_type = "timeout"
        except Exception as e:
            error_message = str(e)
            failure_type = "runtime_error"
        retry_count += 1
        if retry_count <= max_eval_retries:
            logger.debug("Gen %d, Individual %d: evaluation failed, retrying (%d/%d): %s", 
                        current_gen, individual_id, retry_count, max_eval_retries, error_message)
        elif retry_count > max_eval_retries:
            logger.debug("Gen %d, Individual %d: evaluation failed, ran out of retries: %s", 
                        current_gen, individual_id, error_message)
    
    return score, current_program, failure_type


def generate_single_individual(
    individual_id: int, 
    parent_data: Tuple[dict, List[dict]], 
    current_gen: int,
    cfg: Config,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Generate a single individual for the population using pre-sampled data.
    
    This function is separated from the main controller to avoid SQLite connection 
    pickling issues when using ProcessPoolExecutor. It contains all the logic needed
    to generate and evaluate a single individual in isolation.
    
    Returns:
        Dict:
        - score: float | None - fitness score if successful, None if failed
        - program: str - the program code (parent code if failed)
        - explanation: str - LLM's explanation of changes (empty if failed)
        - parent_id: int - ID of the parent individual
        - failure_type: str | None - type of failure if unsuccessful, None if successful
    """
    parent_row, inspiration_rows = parent_data
    
    # Create instances for this individual to avoid conflicts
    individual_llm = OpenAIEngine(cfg.llm)
    patcher = PatchApplier()
    problem = Problem(cfg.problem_entry, cfg.problem_eval)
    prompt_sampler = PromptSampler(None)  # No database needed for prompt building
    
    # Initial prompt
    prompt = prompt_sampler.build(parent_row, inspiration_rows)
    initial_response = individual_llm.generate(prompt)
    
    # Parse initial response to get explanation and code
    explanation, code_section = parse_structured_response(initial_response)
    if explanation is None or code_section is None:
        raise ValueError("Invalid initial response")
    
    # Apply patch with retries
    child_program = apply_patch_with_retries(
        parent_row, code_section, individual_llm, individual_id, 
        current_gen, cfg.evolution.max_patch_retries, prompt_sampler, patcher, logger
    )
    
    if not child_program:
        logger.debug("Gen %d, Individual %d: invalid patch, skipping", current_gen, individual_id)
        return {
            "score": None,
            "explanation": explanation,
            "program": None,
            "parent_id": parent_row["id"],
            "failure_type": "patch_failure"
        }

    # Evaluate with retries
    score, final_program, failure_type = evaluate_with_retries(
        child_program, individual_llm, individual_id, current_gen,
        cfg.evolution.max_eval_retries, prompt_sampler, patcher, problem, 
        cfg.evolution.evaluation_timeout, logger
    )
    
    if score is None:
        logger.debug("Gen %d, Individual %d: evaluation failed (%s), skipping", 
                    current_gen, individual_id, failure_type or "unknown")
        return {
            "score": None,
            "explanation": explanation,
            "program": final_program,
            "parent_id": parent_row["id"],
            "failure_type": failure_type or "evaluation_failure"
        }

    # Success
    logger.debug("Gen %d, Individual %d: new score %.3f", current_gen, individual_id, score)
    return {
        "score": score,
        "program": final_program,
        "explanation": explanation,
        "parent_id": parent_row["id"],
        "failure_type": None
    }