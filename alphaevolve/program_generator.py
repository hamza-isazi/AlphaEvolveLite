import tempfile
import logging
import time
import random
from concurrent.futures import TimeoutError
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .patcher import PatchApplier, PatchError
from .prompts import PromptSampler
from .problem import Problem
from .llm import LLMEngine
from openai import OpenAI
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
    eval_timeout: float
    logger: logging.Logger


def create_program_generation_context(cfg: Config, logger: logging.Logger, client: OpenAI) -> ProgramGenerationContext:
    """Create a program generation context with all necessary dependencies."""    
    return ProgramGenerationContext(
        llm_instance=LLMEngine(cfg.llm, client, logger),
        patcher=PatchApplier(),
        problem=Problem(cfg.problem_entry, cfg.problem_eval),
        prompt_sampler=PromptSampler(None, enable_feedback=cfg.evolution.enable_feedback),  # No database needed for prompt building
        max_retries=cfg.evolution.max_retries,
        eval_timeout=cfg.evolution.eval_timeout,
        logger=logger
    )


def generate_initial_response(context: ProgramGenerationContext, parent_row: dict, inspiration_rows: List[dict], record: ProgramRecord, use_tabu_search: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """Generate the initial response in the form of an explanation and a code diff section from parent and inspiration data."""
    prompt = context.prompt_sampler.build(parent_row, inspiration_rows, use_tabu_search=use_tabu_search)
    initial_response = context.llm_instance.generate(prompt)
    
    # Parse initial response to get explanation and code
    explanation, code_section = parse_structured_response(initial_response)
    
    return explanation, code_section


def handle_program_generation_error(error: Exception) -> Tuple[str, str]:
    """Handle program generation errors and return error message and failure type."""
    if isinstance(error, PatchError):
        return str(error), "patch_failure"
    elif isinstance(error, SyntaxError):
        return f"Syntax error: {str(error)}", "syntax_error"
    elif isinstance(error, TimeoutError):
        return f"Timeout error: {str(error)}", "timeout"
    else:
        return str(error), "runtime_error"


def generate_retry_response(context: ProgramGenerationContext, record: ProgramRecord) -> Optional[str]:
    """Generate a retry response when initial generation fails.
    This is a response to the error message and failure type, and is used to generate a new code diff section."""
    # Select retry model before generating response
    context.llm_instance.select_retry_model()
    
    retry_prompt = context.prompt_sampler.build_retry_prompt(
        record.code, record.error_message, record.failure_type
    )
    
    response = context.llm_instance.generate(retry_prompt)
    
    # Parse the retry response to get new code
    _, code_section = parse_structured_response(response)
    return code_section


def generate_program(
    individual_id: int, 
    parent_data: Tuple[dict, List[dict]], 
    current_gen: int,
    cfg: Config,
    logger: logging.Logger,
    client: OpenAI
) -> ProgramRecord:
    """
    Generate a single program for the population using pre-sampled data.
    
    This function is separated from the main controller to avoid SQLite connection 
    pickling issues when using ProcessPoolExecutor. It contains all the logic needed
    to generate and evaluate a single program in isolation.
    
    Returns:
        ProgramRecord object containing all generation results
    """
    # Create program generation context for this individual with the shared client
    context = create_program_generation_context(cfg, logger, client)
    
    parent_row, inspiration_rows = parent_data
    
    # Create the program record at the start and update it as we go
    generation_start_time = time.time()
    record = ProgramRecord(
        code=parent_row["code"],
        explanation="",
        score=None,
        gen=current_gen,
        parent_id=parent_row["id"],
        used_model=context.llm_instance.get_used_model()
    )
    
    # Decide whether to use tabu search or improvement approach
    use_tabu_search = random.random() < cfg.evolution.tabu_search_probability    
    # Generate initial program
    try:
        explanation, code_diff_section = generate_initial_response(context, parent_row, inspiration_rows, record, use_tabu_search=use_tabu_search)
    except Exception as e:
        error_message, failure_type = handle_program_generation_error(e)
        record.error_message = error_message
        record.failure_type = failure_type
        context.logger.debug("Gen %d, Individual %d: generation failed (%s): %s, model %s", 
                    current_gen, individual_id, record.failure_type, record.error_message, record.used_model)
        return record
    
    if explanation is None or code_diff_section is None:
        record.failure_type = "invalid_response"
        record.error_message = "Invalid initial response from LLM"
        context.logger.debug("Gen %d, Individual %d: generation failed (%s): %s, model %s", 
                    current_gen, individual_id, record.failure_type, record.error_message, record.used_model)
        return record
    record.explanation = explanation

    # Generate and evaluate with retries
    for retry_count in range(context.max_retries + 1):                
        try:
            if retry_count > 0:
                context.logger.debug("Gen %d, Individual %d: %s, retrying (%d/%d): %s", 
                            current_gen, individual_id, record.failure_type, retry_count, context.max_retries, record.error_message)
                
                # Generate a retry response
                code_diff_section = generate_retry_response(context, record)
                if code_diff_section is None:
                    record.failure_type = "invalid_response"
                    record.error_message = "Invalid response from LLM"
                    break

            # Apply the patch to the code
            record.code = context.patcher.apply_diff(record.code, code_diff_section)
            # Compile the new program to check for syntax errors
            compile(record.code, "<candidate>", "exec")
            
            # Write to temp file for evaluation
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as tmp:
                tmp.write(record.code)
                tmp.flush()
                
                # Run evaluation with timeout
                score, execution_time, logs = context.problem.evaluate_with_timeout(tmp.name, context.eval_timeout)
            record.score = score
            record.total_evaluation_time += execution_time
            record.evaluation_logs = logs
            record.failure_type = None
            record.error_message = None
            break
            
        except Exception as e:
            error_message, failure_type = handle_program_generation_error(e)
            record.error_message = error_message
            record.failure_type = failure_type
    
    record.retry_count = retry_count
    record.conversation = context.llm_instance.get_conversation_json()
    
    # If the program failed to generate a valid score, return the record with failure type
    if record.score is None:
        # Get metrics from LLM engine even for failed generations
        record.total_llm_time, record.total_tokens = context.llm_instance.get_metrics()
        context.logger.debug("Gen %d, Individual %d: generation failed (%s): %s, model %s", 
                    current_gen, individual_id, record.failure_type, record.error_message, record.used_model)
        record.generation_time = time.time() - generation_start_time
        return record

    # Success - generate feedback for the successful program if enabled
    if cfg.evolution.enable_feedback:
        # Reset the conversation since we just want feedback on the final program
        context.llm_instance.reset_conversation()
        # Select retry model for feedback generation
        context.llm_instance.select_retry_model()
        try:
            feedback_prompt = context.prompt_sampler.build_feedback_prompt(record.code, record.score, record.evaluation_logs, cfg.problem_eval)
            record.feedback = context.llm_instance.generate(feedback_prompt)
        except Exception as e:
            context.logger.error("Gen %d, Individual %d: failed to generate feedback: %s", 
                                current_gen, individual_id, str(e))
    
    # Get final metrics from LLM engine
    record.total_llm_time, record.total_tokens = context.llm_instance.get_metrics()
    
    # Success - log and return the record
    record.generation_time = time.time() - generation_start_time
    context.logger.debug("Gen %d, Individual %d: new score %.3f, total eval time %.2fs, generation time %.2fs, total LLM time %.2fs, total tokens %d, model %s", 
                current_gen, individual_id, record.score, record.total_evaluation_time, 
                record.generation_time, record.total_llm_time, record.total_tokens, record.used_model)
    return record