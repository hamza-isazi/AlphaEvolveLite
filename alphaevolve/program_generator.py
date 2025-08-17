import tempfile
import logging
import time
import random
from concurrent.futures import TimeoutError
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .patcher import PatchApplier, PatchError
from .prompts.prompt_sampler import PromptSampler
from .problem import Problem
from .llm import LLMEngine, LLMAPIError
from .config import Config
from .response_parser import parse_code_response
from .db import ProgramRecord


def compile_with_context(code: str, filename: str = "<candidate>", num_context_lines: int = 5) -> None:
    """
    Compile code with enhanced error reporting that includes context lines.
    
    Args:
        code: The Python code to compile
        filename: The filename to use for compilation (for error reporting)
        num_context_lines: The number of lines of context to include above and below the error line
    
    Raises:
        SyntaxError: If compilation fails, with enhanced error message including context
    """
    try:
        compile(code, filename, "exec")
    except SyntaxError as e:
        # Extract line number from the error
        if hasattr(e, 'lineno') and e.lineno is not None:
            line_num = e.lineno
            lines = code.split('\n')
            
            # Calculate context range (5 lines above and below)
            start_line = max(0, line_num - num_context_lines - 1)  # -6 because line numbers are 1-indexed
            end_line = min(len(lines), line_num + num_context_lines)
            
            # Build context message
            context_lines = []
            for i in range(start_line, end_line):
                line_content = lines[i] if i < len(lines) else ""
                line_num_display = i + 1
                
                if line_num_display == line_num:
                    # Mark the error line with an arrow
                    context_lines.append(f"{line_num_display:4d} >>> {line_content}")
                else:
                    context_lines.append(f"{line_num_display:4d}     {line_content}")
            
            context_message = "\n".join(context_lines)
            
            # Create enhanced error message
            enhanced_msg = f"Syntax error at line {line_num}: {str(e.msg)}\n{context_message}"
            
            # Create new SyntaxError with enhanced message
            new_error = SyntaxError(enhanced_msg)
            raise new_error
        else:
            # If we can't extract line number, just re-raise the original error
            raise


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


def create_program_generation_context(cfg: Config, logger: logging.Logger) -> ProgramGenerationContext:
    """Create a program generation context with all necessary dependencies."""    
    return ProgramGenerationContext(
        llm_instance=LLMEngine(cfg.llm, logger),
        patcher=PatchApplier(),
        problem=Problem(cfg.problem_entry, cfg.problem_eval),
        prompt_sampler=PromptSampler(None, enable_feedback=cfg.evolution.enable_feedback),  # No database needed for prompt building
        max_retries=cfg.evolution.max_retries,
        eval_timeout=cfg.evolution.eval_timeout,
        logger=logger
    )


def generate_initial_response(context: ProgramGenerationContext, parent_row: dict, inspiration_rows: List[dict], record: ProgramRecord, use_tabu_search: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """Generate the initial response in the form of an explanation and a code diff section from parent and inspiration data."""
    prompt = context.prompt_sampler.build_initial_prompt(parent_row, inspiration_rows, use_tabu_search=use_tabu_search)
    initial_response = context.llm_instance.generate(prompt)
    # Parse initial response to get explanation and code
    explanation, code_section = parse_code_response(initial_response)    
    return explanation, code_section


def handle_program_generation_error(error: Exception) -> Tuple[str, str]:
    """Handle program generation errors and return error message and failure type."""
    if isinstance(error, PatchError):
        return str(error), "patch_failure"
    elif isinstance(error, SyntaxError):
        return f"Syntax error: {str(error)}", "syntax_error"
    elif isinstance(error, TimeoutError):
        return f"Timeout error: {str(error)}", "timeout"
    elif isinstance(error, LLMAPIError):
        return f"LLM API error: {str(error)}", "llm_api_error"
    else:
        return str(error), "runtime_error"


def generate_retry_response(context: ProgramGenerationContext, record: ProgramRecord) -> Optional[str]:
    """Generate a retry response when initial generation fails.
    This is a response to the error message and failure type, and is used to generate a new code diff section."""
    # Select retry model before generating response
    context.llm_instance.select_retry_model()
    
    retry_prompt = context.prompt_sampler.build_retry_prompt(
        messages=context.llm_instance.messages,
        current_code=record.code,
        error_message=record.error_message,
        failure_type=record.failure_type
    )
    
    response = context.llm_instance.generate(retry_prompt)
    
    # Parse the retry response to get new code
    _, code_response = parse_code_response(response)
    return code_response


def generate_program(
    individual_id: int, 
    parent_data: Tuple[dict, List[dict]], 
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
    # Create program generation context for this individual
    context = create_program_generation_context(cfg, logger)
    
    parent_row, inspiration_rows = parent_data
    
    # Create the program record at the start and update it as we go
    generation_start_time = time.time()
    record = ProgramRecord(
        code=parent_row["code"],
        explanation="",
        score=None,
        parent_id=parent_row["id"],
        used_model=context.llm_instance.get_used_model()
    )
    
    # Decide whether to use tabu search or improvement approach
    use_tabu_search = random.random() < cfg.evolution.tabu_search_probability    
    # Generate initial program
    try:
        explanation, code_response = generate_initial_response(context, parent_row, inspiration_rows, record, use_tabu_search=use_tabu_search)
        record.conversation = context.llm_instance.get_conversation_json()
    except Exception as e:
        # If we don't even get a valid initial response, we just return the record with the error message
        # as there's no valid program to fix
        error_message, failure_type = handle_program_generation_error(e)
        record.error_message = error_message
        record.failure_type = failure_type
        record.generation_time = time.time() - generation_start_time
        record.total_llm_time, record.total_tokens = context.llm_instance.get_metrics()
        return record
    record.explanation = explanation

    # Generate and evaluate with retries
    for retry_count in range(context.max_retries + 1):                
        try:
            if retry_count > 0:
                context.logger.debug("Individual %d: %s, retrying (%d/%d): %s", 
                            individual_id, record.failure_type, retry_count, context.max_retries, record.error_message)
                
                # Generate a retry response
                code_response = generate_retry_response(context, record)

            # Apply the patch and try to compile it to check for syntax errors
            record.code = context.patcher.apply_diff(record.code, code_response)
            compile_with_context(record.code)
            
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
            # We don't want to retry program generation for LLM API errors (after already retrying the API call with exponential backoff)
            if failure_type == "llm_api_error":
                break
    
    record.retry_count = retry_count
    record.conversation = context.llm_instance.get_conversation_json()
    
    # If the program failed to generate a valid score, return the record with failure type
    if record.score is None:
        # Get metrics from LLM engine even for failed generations
        record.total_llm_time, record.total_tokens = context.llm_instance.get_metrics()
        record.generation_time = time.time() - generation_start_time
        return record

    # Success - generate feedback for the successful program if enabled
    if cfg.evolution.enable_feedback:
        # Reset the conversation since we just want feedback on the final program
        context.llm_instance.reset_conversation()
        # Select retry model for feedback generation
        context.llm_instance.select_retry_model()
        try:
            feedback_prompt = context.prompt_sampler.build_feedback_prompt(record.code, record.score, record.evaluation_logs)
            record.feedback = context.llm_instance.generate(feedback_prompt)
        except Exception as e:
            context.logger.error("Individual %d: failed to generate feedback: %s", 
                                individual_id, str(e))
    
    # Get final metrics
    record.total_llm_time, record.total_tokens = context.llm_instance.get_metrics()
    record.generation_time = time.time() - generation_start_time
    return record