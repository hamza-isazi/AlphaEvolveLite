import tempfile
import logging
import time
from concurrent.futures import TimeoutError
from typing import List, Tuple
from dataclasses import dataclass

from .patcher import PatchApplier, PatchError
from .prompts import PromptSampler
from .problem import Problem
from .llm import LLMEngine
from openai import OpenAI
from .config import Config
from .response_parser import parse_structured_response
from .db import ProgramRecord

FEEDBACK_PROMPT = """\
You are an expert code reviewer. Analyze the following program and provide concise feedback on potential issues or areas for improvement.

**Program Code:**
```python
{code}
```

**Program Score:** {score}

**Evaluation Logs:**
{logs}

**Task:** Provide 2-3 concise, critical, and constructive feedback on this program. Focus on:
- Performance issues
- Code quality problems
- Logic errors or edge cases
- Potential optimizations

Keep your feedback brief and specific. Each suggestion should be 1-2 sentences maximum."""

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


def create_program_generation_context(cfg: Config, logger: logging.Logger, client: OpenAI) -> ProgramGenerationContext:
    """Create a program generation context with all necessary dependencies."""    
    return ProgramGenerationContext(
        llm_instance=LLMEngine(cfg.llm, client),
        patcher=PatchApplier(),
        problem=Problem(cfg.problem_entry, cfg.problem_eval),
        prompt_sampler=PromptSampler(None, enable_feedback=cfg.evolution.enable_feedback),  # No database needed for prompt building
        max_retries=cfg.evolution.max_retries,
        evaluation_timeout=cfg.evolution.evaluation_timeout,
        logger=logger
    )


def generate_feedback(code: str, score: float, logs: str, llm_instance: LLMEngine) -> str:
    """
    Generate feedback for a successful program by resetting the conversation
    and asking for concise feedback on issues with the program.
    
    Args:
        code: The program code
        score: The program's score
        logs: The evaluation logs
        llm_instance: The LLM engine instance
        
    Returns:
        Generated feedback as a string
    """
    # Reset the conversation to start fresh
    llm_instance.reset_conversation()
    
    # Build the feedback prompt
    feedback_prompt = FEEDBACK_PROMPT.format(
        code=code,
        score=f"{score:.3f}",
        logs=logs if logs else "No evaluation logs available."
    )
    
    # Generate feedback
    feedback_response, _, _ = llm_instance.generate(feedback_prompt)
    
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
    
    # Track total generation time
    generation_start_time = time.time()
    
    # Track total LLM time, total evaluation time, total tokens, and evaluation logs
    total_llm_time = 0.0
    total_evaluation_time = 0.0
    total_tokens = 0
    logs = None
    
    # Initial prompt
    prompt = context.prompt_sampler.build(parent_row, inspiration_rows)
    initial_response, initial_response_time, initial_tokens = context.llm_instance.generate(prompt)
    total_llm_time += initial_response_time
    total_tokens += initial_tokens
    
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
            
            response, response_time, response_tokens = context.llm_instance.generate(retry_prompt)
            total_llm_time += response_time
            total_tokens += response_tokens
            
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
                score, execution_time, logs = context.problem.evaluate_with_timeout(tmp.name, context.evaluation_timeout)
                total_evaluation_time += execution_time
                
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
        
        if retry_count < context.max_retries:
            retry_count += 1
            context.logger.debug("Gen %d, Individual %d: %s, retrying (%d/%d): %s", 
                        current_gen, individual_id, failure_type, retry_count, context.max_retries, error_message)
        else:
            context.logger.debug("Gen %d, Individual %d: %s, ran out of retries: %s", 
                        current_gen, individual_id, failure_type, error_message)
            break
    
    # Calculate total generation time
    generation_time = time.time() - generation_start_time
    
    # Capture the full conversation history
    conversation_json = context.llm_instance.get_conversation_json()
    
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
            failure_type=failure_type or "generation_failure",
            retry_count=retry_count,
            total_evaluation_time=total_evaluation_time,  # Total evaluation time across all attempts
            generation_time=generation_time,
            total_llm_time=total_llm_time,
            total_tokens=total_tokens,
            conversation=conversation_json,
            evaluation_logs=logs
        )

    # Success - generate feedback for the successful program if enabled
    feedback = None
    if cfg.evolution.enable_feedback:
        feedback = generate_feedback(current_program, score, logs, context.llm_instance)
    
    # Success - return a ProgramRecord with the score and total times
    context.logger.debug("Gen %d, Individual %d: new score %.3f, total eval time %.2fs, generation time %.2fs, total LLM time %.2fs, total tokens %d", 
                current_gen, individual_id, score, total_evaluation_time, generation_time, total_llm_time, total_tokens)
    return ProgramRecord(
        code=current_program,
        explanation=explanation,
        score=score,
        gen=current_gen,
        parent_id=parent_row["id"],
        failure_type=None,
        retry_count=retry_count,
        total_evaluation_time=total_evaluation_time,  # Total evaluation time across all attempts
        generation_time=generation_time,
        total_llm_time=total_llm_time,
        total_tokens=total_tokens,
        conversation=conversation_json,
        evaluation_logs=logs,
        feedback=feedback
    )