import tempfile
import logging
from concurrent.futures import TimeoutError
from typing import List, Tuple, Dict, Any

from .patcher import PatchApplier, PatchError
from .prompts import PromptSampler
from .problem import Problem
from .llm import OpenAIEngine
from .config import Config
from .response_parser import parse_structured_response


def generate_and_evaluate_with_retries(
    parent_row: dict,
    initial_code: str,
    llm_instance: OpenAIEngine,
    individual_id: int,
    current_gen: int,
    max_retries: int,
    prompt_sampler: PromptSampler,
    patcher: PatchApplier,
    problem: Problem,
    evaluation_timeout: float,
    logger: logging.Logger
) -> tuple[float | None, str, str | None]:
    """
    Generate a patch and evaluate it with retry logic for both patch application and evaluation failures.
    
    Returns:
        Tuple of (score, final_program, failure_type)
        - score: float | None - fitness score if successful, None if failed
        - final_program: str - the final program code (parent code if failed)
        - failure_type: str | None - type of failure if unsuccessful, None if successful
    """
    retry_count = 0
    score = None
    error_message = "Unknown error"
    current_program = parent_row["code"]
    failure_type = None
    code_section = initial_code
    
    while retry_count <= max_retries:
        if retry_count > 0:
            retry_prompt = prompt_sampler.build_retry_prompt(
                current_program, error_message, failure_type or "unknown_error"
            )
            
            response = llm_instance.generate(retry_prompt)
            
            # Parse response to get new code
            _, code_section = parse_structured_response(response)
            if code_section is None:
                # If no code found, break out of retry loop
                failure_type = "invalid_response"
                break
        
        try:
            # Apply the patch
            child_program = patcher.apply_diff(current_program, code_section)
            # Compile the new program to check for syntax errors
            compile(child_program, "<candidate>", "exec")
            # If the program compiles, update the current program
            current_program = child_program
            # Write to temp file for evaluator
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as tmp:
                tmp.write(child_program)
                tmp.flush()
                
                # Run evaluation with timeout
                score = problem.evaluate_with_timeout(tmp.name, evaluation_timeout)
                
            # Success - exit retry loop
            failure_type = None
            break
        except PatchError as e:
            error_message = str(e)
            failure_type = "patch_failure"
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
        if retry_count <= max_retries:
            logger.debug("Gen %d, Individual %d: %s, retrying (%d/%d): %s", 
                        current_gen, individual_id, failure_type, retry_count, max_retries, error_message)
        else:
            logger.debug("Gen %d, Individual %d: %s, ran out of retries: %s", 
                        current_gen, individual_id, failure_type, error_message)
    
    return score, current_program, failure_type


def generate_program(
    individual_id: int, 
    parent_data: Tuple[dict, List[dict]], 
    current_gen: int,
    cfg: Config,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Generate a single program for the population using pre-sampled data.
    
    This function is separated from the main controller to avoid SQLite connection 
    pickling issues when using ProcessPoolExecutor. It contains all the logic needed
    to generate and evaluate a single program in isolation.
    
    Returns:
        Dict:
        - score: float | None - fitness score if successful, None if failed
        - program: str - the program code (parent code if failed)
        - explanation: str - LLM's explanation of changes (empty if failed)
        - parent_id: int - ID of the parent program
        - failure_type: str | None - type of failure if unsuccessful, None if successful
    """
    parent_row, inspiration_rows = parent_data
    
    # Create instances for this program to avoid conflicts
    llm_instance = OpenAIEngine(cfg.llm)
    patcher = PatchApplier()
    problem = Problem(cfg.problem_entry, cfg.problem_eval)
    prompt_sampler = PromptSampler(None)  # No database needed for prompt building
    
    # Initial prompt
    prompt = prompt_sampler.build(parent_row, inspiration_rows)
    initial_response = llm_instance.generate(prompt)
    
    # Parse initial response to get explanation and code
    explanation, code_section = parse_structured_response(initial_response)
    if explanation is None or code_section is None:
        raise ValueError("Invalid initial response")
    
    # Generate and evaluate with retries
    score, final_program, failure_type = generate_and_evaluate_with_retries(
        parent_row, code_section, llm_instance, individual_id, current_gen,
        cfg.evolution.max_retries, prompt_sampler, patcher, problem,
        cfg.evolution.evaluation_timeout, logger
    )
    
    if score is None:
        logger.debug("Gen %d, Individual %d: generation failed (%s), skipping", 
                    current_gen, individual_id, failure_type or "unknown")
        return {
            "score": None,
            "explanation": explanation,
            "program": final_program,
            "parent_id": parent_row["id"],
            "failure_type": failure_type or "generation_failure"
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