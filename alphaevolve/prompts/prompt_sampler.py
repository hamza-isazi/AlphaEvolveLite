"""
Jinja-based prompt sampler for AlphaEvolveLite.
"""

from __future__ import annotations
from typing import Sequence, Dict, List, Optional
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from ..patcher import _EVOLVE_RE


class PromptSampler:
    """Builds prompts using Jinja templates."""

    def __init__(self, archive, k_elite=3, k_rand=2, enable_feedback=True):
        self.archive = archive
        self.k_elite = k_elite
        self.k_rand = k_rand
        self.enable_feedback = enable_feedback
        
        # Setup Jinja environment
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Cache templates
        self._templates = {}
    
    def _get_template(self, name: str):
        """Get a template by name with caching."""
        if name not in self._templates:
            self._templates[name] = self.env.get_template(name)
        return self._templates[name]

    def _has_evolve_blocks(self, code: str) -> bool:
        """Check if the code contains evolve blocks."""
        return bool(_EVOLVE_RE.search(code))

    def build_initial_prompt(self, parent_row: dict, inspiration_rows: Sequence[Dict], use_tabu_search: bool = False) -> str:
        """Build the initial evolution prompt using Jinja templates."""
        # Calculate the target score to beat
        all_scores = [parent_row['score']] + [r['score'] for r in inspiration_rows]
        target_score = max(all_scores)
        
        # Check if parent has evolve blocks
        has_evolve_blocks = self._has_evolve_blocks(parent_row['code'])
        
        # Let Jinja handle all the formatting and logic
        template = self._get_template('evolution.jinja')
        return template.render(
            parent=parent_row,
            inspirations=inspiration_rows,
            target_score=target_score,
            has_evolve_blocks=has_evolve_blocks,
            use_tabu_search=use_tabu_search,
            enable_feedback=self.enable_feedback
        )

    def build_retry_prompt(self, messages: List[Dict], current_code: str, error_message: str, failure_type: str) -> str:
        """Build a retry prompt using Jinja templates."""
        # Check if code is unchanged in conversation
        code_unchanged = any(msg["role"] == "user" and current_code in msg["content"] for msg in messages)
        
        template = self._get_template('retry.jinja')
        return template.render(
            code_unchanged=code_unchanged,
            failure_type=failure_type,
            error_message=error_message,
            current_code=current_code
        )
    
    def build_feedback_prompt(self, code: str, score: float, logs: str, evaluation_script_path: str = None) -> str:
        """Build a feedback prompt using Jinja templates."""
        # Read evaluation script if provided
        evaluation_script = "No evaluation script available."
        if evaluation_script_path:
            try:
                with open(evaluation_script_path, 'r', encoding='utf-8') as f:
                    evaluation_script = f.read()
            except Exception as e:
                evaluation_script = f"Error reading evaluation script: {str(e)}"
        
        template = self._get_template('feedback.jinja')
        return template.render(
            code=code,
            score=score,
            evaluation_script=evaluation_script,
            logs=logs if logs else "No evaluation logs available."
        )
