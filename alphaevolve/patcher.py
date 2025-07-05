"""
Apply one-or-many AlphaEvolve SEARCH/REPLACE diff blocks.
"""

import re
from typing import Optional


_BLOCK_RE = re.compile(
    r"<{3,}\s*SEARCH\s*(.*?)\s*={3,}\s*(.*?)\s*>{3,}\s*REPLACE",
    re.DOTALL,
)


class PatchApplier:
    @staticmethod
    def apply_diff(source: str, diff_text: str) -> Optional[str]:
        """
        Apply all SEARCH/REPLACE blocks to `source`.
        Return new source or None if any SEARCH part not found exactly.
        """
        new = source
        for search, replace in _BLOCK_RE.findall(diff_text):
            if search not in new:
                # LLM produced a search snippet that doesn't match – abort
                return None
            new = new.replace(search, replace, 1)
        return new if new != source else None  # None → no change

    @staticmethod
    def is_valid(py_code: str) -> bool:
        try:
            compile(py_code, "<candidate>", "exec")
            return True
        except SyntaxError:
            return False