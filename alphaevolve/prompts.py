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
    on the task at hand:

    {inspirations}

    # Current program

    Here is the current program we are trying to improve (you will
    need to propose a modification to it below).

    {parent}

    # Task
    Suggest improvements to the program that will lead to better performance on the specified metrics.

    You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:
    ```
    <<<<<<< SEARCH
    # original lines (one or more consecutive lines)
    =======
    # replacement lines (same number of lines or valid replacement)
    >>>>>>> REPLACE
    ```

    Example of valid diff format:
    ```
    <<<<<<< SEARCH
    poem stub
    =======
    Tyger Tyger, burning bright, In the forests of the night; What immortal hand or eye
    >>>>>>> REPLACE
    ```

    ### SEARCH/REPLACE block rules
    {evolve_instructions}
    - Emit *each* independent modification as its own complete block
    - Do **not** nest multiple SEARCH/REPLACE blocks together—each change must start with `<<<<<<< SEARCH` and end with `>>>>>>> REPLACE`.
    - You can suggest multiple changes, they will be applied in order.
    - Each SEARCH section must match the code EXACTLY, including all whitespace, indentation, and newlines.
    - SEARCH/REPLACE blocks must NOT overlap - each must target different, non-overlapping code sections
    - Be thoughtful about your changes and explain your reasoning thoroughly.
    - Make sure that the changes you propose are consistent with each
    other. For example, if you refer to a new config variable
    somewhere, you should also propose a change to add that
    variable.
    """

EVOLVE_INSTRUCTIONS = """\
    - Only change lines *between* the markers  
    `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`.  
    - Never alter code outside an EVOLVE block; any such hunk is rejected."""

FREE_INSTRUCTIONS = """\
    - You can modify any part of the code as needed."""


class PromptSampler:
    """Builds a single prompt each generation."""

    def __init__(self, archive, k_elite=3, k_rand=2):
        self.archive = archive
        self.k_elite = k_elite
        self.k_rand = k_rand
        
    @staticmethod
    def _format_rows(rows: Sequence[Dict]) -> str:
        out = []
        for r in rows:
            out.append(f"Score {r['score']:.3f}:\n```\n{r['code']}\n```")
        return "\n\n".join(out) if out else "None yet."

    @staticmethod
    def _has_evolve_blocks(code: str) -> bool:
        """Check if the code contains evolve blocks."""
        return bool(_EVOLVE_RE.search(code))

    def build(self, parent_row: dict, inspiration_rows: Sequence[Dict]) -> str:
        # Choose evolve instructions based on whether evolve blocks are present
        evolve_instructions = EVOLVE_INSTRUCTIONS if self._has_evolve_blocks(parent_row['code']) else FREE_INSTRUCTIONS
            
        prompt = TEMPLATE.format(
            parent=self._format_rows([parent_row]),
            inspirations=self._format_rows(inspiration_rows),
            evolve_instructions=evolve_instructions,
        )
        return prompt