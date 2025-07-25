import tempfile
import logging
import time
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
        llm_instance=LLMEngine(cfg.llm, client),
        patcher=PatchApplier(),
        problem=Problem(cfg.problem_entry, cfg.problem_eval),
        prompt_sampler=PromptSampler(None, enable_feedback=cfg.evolution.enable_feedback),  # No database needed for prompt building
        max_retries=cfg.evolution.max_retries,
        eval_timeout=cfg.evolution.eval_timeout,
        logger=logger
    )


def generate_initial_response(context: ProgramGenerationContext, parent_row: dict, inspiration_rows: List[dict], record: ProgramRecord) -> Tuple[Optional[str], Optional[str]]:
    """Generate the initial response in the form of an explanation and a code diff section from parent and inspiration data."""
    prompt = context.prompt_sampler.build(parent_row, inspiration_rows)
    initial_response, initial_response_time, initial_tokens = context.llm_instance.generate(prompt)
    
    # Update record metrics
    record.total_llm_time += initial_response_time
    record.total_tokens += initial_tokens
    
    # Parse initial response to get explanation and code
    explanation, code_section = parse_structured_response(initial_response)
    
    return explanation, code_section


def handle_evaluation_error(error: Exception, context: ProgramGenerationContext) -> Tuple[str, str]:
    """Handle evaluation errors and return error message and failure type."""
    if isinstance(error, PatchError):
        return str(error), "patch_failure"
    elif isinstance(error, SyntaxError):
        return f"Syntax error: {str(error)}", "syntax_error"
    elif isinstance(error, TimeoutError):
        return f"Evaluation timed out after {context.eval_timeout} seconds", "timeout"
    else:
        return str(error), "runtime_error"


def generate_retry_response(context: ProgramGenerationContext, record: ProgramRecord) -> Optional[str]:
    """Generate a retry response when initial generation fails.
    This is a response to the error message and failure type, and is used to generate a new code diff section."""
    retry_prompt = context.prompt_sampler.build_retry_prompt(
        record.code, record.error_message, record.failure_type
    )
    
    response, response_time, response_tokens = context.llm_instance.generate(retry_prompt)
    record.total_llm_time += response_time
    record.total_tokens += response_tokens
    
    # Parse the retry response to get new code
    _, code_section = parse_structured_response(response)
    return code_section


def generate_feedback(record: ProgramRecord, ctx: ProgramGenerationContext, evaluation_script_path: str = None) -> str:
    """
    Generate feedback for a program by asking for insights into why the program achieved its specific score.
    
    Args:
        record: The program record
        ctx: The program generation context
        evaluation_script_path: Path to the evaluation script
        
    Returns:
        Generated feedback as a string
    """
    # Create a prompt sampler to build the feedback prompt
    feedback_prompt = ctx.prompt_sampler.build_feedback_prompt(record.code, record.score, record.evaluation_logs, evaluation_script_path)
    
    # Generate feedback
    feedback_response, response_time, response_tokens = ctx.llm_instance.generate(feedback_prompt)
    record.total_llm_time += response_time
    record.total_tokens += response_tokens
    
    return feedback_response.strip() if feedback_response else "No feedback generated."


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
        parent_id=parent_row["id"]
    )
    
    # Generate initial program
    explanation, code_diff_section = generate_initial_response(context, parent_row, inspiration_rows, record)
    if explanation is None or code_diff_section is None:
        record.failure_type = "invalid_response"
        record.error_message = "Invalid initial response from LLM"
        context.logger.debug("Gen %d, Individual %d: generation failed (%s): %s", 
                    current_gen, individual_id, record.failure_type, record.error_message)
        record.generation_time = time.time() - generation_start_time
        return record
    record.explanation = explanation

    # Generate and evaluate with retries
    for retry_count in range(context.max_retries + 1):        
        if retry_count > 0:
            context.logger.debug("Gen %d, Individual %d: %s, retrying (%d/%d): %s", 
                        current_gen, individual_id, record.failure_type, retry_count, context.max_retries, record.  error_message)
            
            # Generate a retry response
            code_diff_section = generate_retry_response(context, record)
            if code_diff_section is None:
                record.failure_type = "invalid_response"
                record.error_message = "Invalid response from LLM"
                break
        
        try:
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
            error_message, failure_type = handle_evaluation_error(e, context)
            record.error_message = error_message
            record.failure_type = failure_type
    
    record.retry_count = retry_count
    record.conversation = context.llm_instance.get_conversation_json()
    
    # If the program failed to generate a valid score, return the record with failure type
    if record.score is None:
        context.logger.debug("Gen %d, Individual %d: generation failed (%s): %s", 
                    current_gen, individual_id, record.failure_type, record.error_message)
        record.generation_time = time.time() - generation_start_time
        return record

    # Success - generate feedback for the successful program if enabled
    if cfg.evolution.enable_feedback:
        # Reset the conversation since we just want feedback on the final program
        context.llm_instance.reset_conversation()
        record.feedback = generate_feedback(record, context, cfg.problem_eval)
    
    # Success - log and return the record
    record.generation_time = time.time() - generation_start_time
    context.logger.debug("Gen %d, Individual %d: new score %.3f, total eval time %.2fs, generation time %.2fs, total LLM time %.2fs, total tokens %d", 
                current_gen, individual_id, record.score, record.total_evaluation_time, 
                record.generation_time, record.total_llm_time, record.total_tokens)
    return record