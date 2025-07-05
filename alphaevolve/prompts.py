"""
Prompt assembler that mimics AlphaEvolve’s SEARCH/REPLACE style.
"""

from __future__ import annotations
from typing import Sequence, Dict
import textwrap

TEMPLATE = """\
    Act as an expert software developer. Your task is to iteratively improve the provided codebase.

    - Prior programs

    Previously we found that the following programs performed well
    on the task at hand:

    {inspirations}

    - Current program

    Here is the current program we are trying to improve (you will
    need to propose a modification to it below).

    {parent}

    Follow the SEARCH/REPLACE block rules described below.

    SEARCH/REPLACE block rules:
        <<<<<<< SEARCH
        # original code to match
        =======
        # new code to replace it with
        >>>>>>> REPLACE

    Make sure that the changes you propose are consistent with each
    other. For example, if you refer to a new config variable
    somewhere, you should also propose a change to add that
    variable.

    Task
    Suggest a new idea to improve the code that is inspired by your
    expert knowledge of optimization and machine learning.

    Describe each change with a SEARCH/REPLACE block.
    """


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

    def build(self, parent_row: dict, inspiration_rows: Sequence[Dict]) -> str:
        prompt = TEMPLATE.format(
            parent=self._format_rows([parent_row]),
            inspirations=self._format_rows(inspiration_rows),
        )
        return prompt