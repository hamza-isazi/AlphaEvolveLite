import tempfile
import logging
from concurrent.futures import TimeoutError
from typing import List, Tuple
from dataclasses import dataclass

from .patcher import PatchApplier, PatchError
from .prompts import PromptSampler
from .problem import Problem
from .llm import LLMEngine, create_llm_engine
from .config import Config
from .response_parser import parse_structured_response
from .db import ProgramRecord

@dataclass
class ProgramGenerationContext:
    """Context object containing all dependencies needed for program generation."""
    llm_instance: LLMEngine
    patcher: PatchApplier
    problem: Problem
    prompt_sampler: PromptSampler
    max_retries: int
    evaluation_timeout: float
    logger: logging.Logger


def create_program_generation_context(cfg: Config, logger: logging.Logger) -> ProgramGenerationContext:
    """Create a program generation context with all necessary dependencies."""
    return ProgramGenerationContext(
        llm_instance=create_llm_engine(cfg.llm),
        patcher=PatchApplier(),
        problem=Problem(cfg.problem_entry, cfg.problem_eval),
        prompt_sampler=PromptSampler(None),  # No database needed for prompt building
        max_retries=cfg.evolution.max_retries,
        evaluation_timeout=cfg.evolution.evaluation_timeout,
        logger=logger
    )


def generate_program(
    individual_id: int, 
    parent_data: Tuple[dict, List[dict]], 
    current_gen: int,
    cfg: Config,
    logger: logging.Logger
) -> ProgramRecord:
    """
    Generate a single program for the population using pre-sampled data.
    
    This function is separated from the main controller to avoid SQLite connection 
    pickling issues when using ProcessPoolExecutor. It contains all the logic needed
    to generate and evaluate a single program in isolation.
    
    Returns:
        ProgramRecord object containing all generation results
    """
    # Create program generation context for this individual to avoid sharing LLM instances
    context = create_program_generation_context(cfg, logger)
    
    parent_row, inspiration_rows = parent_data
    
    # Initial prompt
    prompt = context.prompt_sampler.build(parent_row, inspiration_rows)
    initial_response = context.llm_instance.generate(prompt)
    
    # Parse initial response to get explanation and code
    explanation, code_section = parse_structured_response(initial_response)
    if explanation is None or code_section is None:
        raise ValueError("Invalid initial response")
    
    # Generate and evaluate with retries
    retry_count = 0
    score = None
    error_message = "Unknown error"
    current_program = parent_row["code"]
    failure_type = None
    
    # Main retry loop
    while retry_count <= context.max_retries:
        if retry_count > 0:
            retry_prompt = context.prompt_sampler.build_retry_prompt(
                current_program, error_message, failure_type or "unknown_error"
            )
            
            response = context.llm_instance.generate(retry_prompt)
            
            # Parse response to get new code
            _, code_section = parse_structured_response(response)
            if code_section is None:
                # If no code found, break out of retry loop
                failure_type = "invalid_response"
                break
        
        try:
            # Apply the patch
            child_program = context.patcher.apply_diff(current_program, code_section)
            # Compile the new program to check for syntax errors
            compile(child_program, "<candidate>", "exec")
            # If the program compiles, update the current program
            current_program = child_program
            # Write to temp file for evaluator
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as tmp:
                tmp.write(child_program)
                tmp.flush()
                
                # Run evaluation with timeout
                score = context.problem.evaluate_with_timeout(tmp.name, context.evaluation_timeout)
                
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
            error_message = f"Evaluation timed out after {context.evaluation_timeout} seconds"
            failure_type = "timeout"
        except Exception as e:
            error_message = str(e)
            failure_type = "runtime_error"
        
        retry_count += 1
        if retry_count <= context.max_retries:
            context.logger.debug("Gen %d, Individual %d: %s, retrying (%d/%d): %s", 
                        current_gen, individual_id, failure_type, retry_count, context.max_retries, error_message)
        else:
            context.logger.debug("Gen %d, Individual %d: %s, ran out of retries: %s", 
                        current_gen, individual_id, failure_type, error_message)
    
    # If the program failed to generate a valid score, return a ProgramRecord with the failure type
    if score is None:
        context.logger.debug("Gen %d, Individual %d: generation failed (%s), skipping", 
                    current_gen, individual_id, failure_type or "unknown")
        return ProgramRecord(
            code=current_program,
            explanation=explanation,
            score=None,
            gen=current_gen,
            parent_id=parent_row["id"],
            failure_type=failure_type or "generation_failure"
        )

    # Success - return a ProgramRecord with the score
    context.logger.debug("Gen %d, Individual %d: new score %.3f", current_gen, individual_id, score)
    return ProgramRecord(
        code=current_program,
        explanation=explanation,
        score=score,
        gen=current_gen,
        parent_id=parent_row["id"],
        failure_type=None
    )