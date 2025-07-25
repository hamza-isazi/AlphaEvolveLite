"""
Prompt assembler that mimics AlphaEvolve's SEARCH/REPLACE style.
"""

from __future__ import annotations
from typing import Sequence, Dict
from .patcher import _EVOLVE_RE

TEMPLATE = """\
    Act as an expert software developer. Your task is to iteratively improve the provided codebase.

    # Prior programs

    Previously we found that the following programs performed well
    on the task at hand. Each includes the developer's explanation of their improvements and feedback:

    {inspirations}

    # Current program

    Here is the current program we are trying to improve (you will
    need to propose a modification to it below). It includes the developer's explanation of their improvements and feedback:

    {parent}

    # Task
    Suggest improvements to the program that will lead to better performance on the specified metrics.

    # Response Format
    Your response MUST follow this exact structure:

    ### Explanation
    Briefly describe what you changed, why it helps, and what effect it's expected to have. Keep it under 3 sentences. 
    Do not restate or refer to the current program. Avoid implementation details already visible in the code diff.

    ### Code
    [Provide your code changes using one of the following formats]
    *Do not explain changes inside the code section.*

    ## Option 1: Major Structural Changes (Full File Replacement)
    If the changes require major structural modifications, output a complete file wrapped in a Markdown code block:
    ```[language]
    [Complete file content here]
    ```

    ## Option 2: Targeted Improvements (SEARCH/REPLACE format)
    If making targeted improvements, output code changes using the SEARCH/REPLACE format:
    ```
    <<<<<<< SEARCH
    # original lines
    =======
    # replacement lines
    >>>>>>> REPLACE

    <<<<<<< SEARCH
    # other set of original lines
    =======
    # other set of replacement lines
    >>>>>>> REPLACE
    ```

    If using the SEARCH/REPLACE format, please follow these rules:
    ### SEARCH/REPLACE block rules
    {evolve_instructions}
    - You can suggest multiple changes, they will be applied in order.
    - Emit *each* independent modification as its own complete SEARCH/REPLACE block, each change must start with `<<<<<<< SEARCH` and end with `>>>>>>> REPLACE`.
    - Each SEARCH section must match the code EXACTLY, including all whitespace, indentation, and newlines.
    - SEARCH/REPLACE blocks must NOT overlap - each must target different, non-overlapping code sections.
    - Make sure that the changes you propose are consistent with each other. For example, if you refer to a new config variable
        somewhere, you should also propose a change to add that variable.
"""

RETRY_TEMPLATE = """\
The previous attempt failed with the following error:

{error_message}

This indicates a {failure_type} issue that needs to be fixed.

Below is the current version of the code that needs to be corrected:

{current_code}

Please generate a new response that fixes the issue. 
The response should be relative to the current code shown above, and not to any earlier version.

Please provide corrected code changes using SEARCH/REPLACE or Full File Replacement format:"""

EVOLVE_INSTRUCTIONS = """\
    - Only change lines *between* the markers  
    `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`.  
    - Never alter code outside an EVOLVE block; any such hunk is rejected."""

FREE_INSTRUCTIONS = """\
    - You can modify any part of the code as needed."""

FEEDBACK_PROMPT = """\
You are an expert evaluator. Analyze the following program's performance and explain why it achieved the score it did.

**Program Code:**
```python
{code}
```

**Program Score:** {score}

**Evaluation Script:**
```python
{evaluation_script}
```

**Evaluation Logs:**
{logs}

**Task:** Provide 2-3 concise insights explaining why this program achieved its specific score. Focus on:
- What the evaluation script is measuring and how the program performed on each metric
- Specific test cases or criteria that the program passed or failed
- Performance characteristics that contributed to the score (speed, accuracy, etc.)
- Any constraints or requirements that the program met or violated

Your analysis should explain the score, not suggest improvements. The goal is to understand what worked and what didn't based on the evaluation results.

Keep your analysis brief and specific. Each insight should be 1-2 sentences maximum."""





class PromptSampler:
    """Builds a single prompt each generation."""

    def __init__(self, archive, k_elite=3, k_rand=2, enable_feedback=True):
        self.archive = archive
        self.k_elite = k_elite
        self.k_rand = k_rand
        self.enable_feedback = enable_feedback
        
    @staticmethod
    def _format_rows(rows: Sequence[Dict], include_feedback: bool = True) -> str:
        out = []
        for r in rows:
            explanation = r.get('explanation', 'No explanation provided')
            feedback = r.get('feedback')
            
            program_text = f"Score {r['score']:.3f}:\nExplanation: {explanation}\n```\n{r['code']}\n```"
            
            if include_feedback and feedback:
                program_text += f"\n\nFeedback:\n{feedback}"
            
            out.append(program_text)
        return "\n\n".join(out) if out else "None yet."

    @staticmethod
    def _format_evaluation_logs(parent_row: Dict) -> str:
        """Format evaluation logs for the parent program."""
        logs = parent_row.get('evaluation_logs')
        if logs:
            return f"```\n{logs}\n```"
        else:
            return "No evaluation logs available."

    @staticmethod
    def _has_evolve_blocks(code: str) -> bool:
        """Check if the code contains evolve blocks."""
        return bool(_EVOLVE_RE.search(code))

    def build(self, parent_row: dict, inspiration_rows: Sequence[Dict]) -> str:
        # Choose evolve instructions based on whether evolve blocks are present
        evolve_instructions = EVOLVE_INSTRUCTIONS if self._has_evolve_blocks(parent_row['code']) else FREE_INSTRUCTIONS
        
        prompt = TEMPLATE.format(
            parent=self._format_rows([parent_row], include_feedback=self.enable_feedback),
            inspirations=self._format_rows(inspiration_rows, include_feedback=self.enable_feedback) if inspiration_rows else "None yet.",
            evolve_instructions=evolve_instructions
        )
        return prompt

    def build_retry_prompt(self, current_code: str, error_message: str, failure_type: str) -> str:
        """Build a unified retry prompt for both patch and evaluation failures."""        
        return RETRY_TEMPLATE.format(
            current_code=f"```\n{current_code}\n```",
            error_message=error_message,
            failure_type=failure_type
        )
    
    def build_feedback_prompt(self, code: str, score: float, logs: str, evaluation_script_path: str = None) -> str:
        """Build a feedback prompt for analyzing program performance."""
        # Read the evaluation script if provided
        evaluation_script_content = "No evaluation script available."
        if evaluation_script_path:
            try:
                with open(evaluation_script_path, 'r', encoding='utf-8') as f:
                    evaluation_script_content = f.read()
            except Exception as e:
                evaluation_script_content = f"Error reading evaluation script: {str(e)}"
        
        # Build the feedback prompt
        return FEEDBACK_PROMPT.format(
            code=code,
            score=f"{score:.3f}",
            evaluation_script=evaluation_script_content,
            logs=logs if logs else "No evaluation logs available."
        )