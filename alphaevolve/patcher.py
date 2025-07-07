import re
from typing import Optional, List, Tuple

_DIFF_RE = re.compile(
    r"<<{4,}\s*SEARCH\s*\n(.*?)\n={4,}\s*\n(.*?)\n>{4,}\s*REPLACE",
    re.DOTALL,
)
_EVOLVE_RE = re.compile(
    r"#\s*EVOLVE-BLOCK-START(.*?)#\s*EVOLVE-BLOCK-END",
    re.DOTALL,
)

class PatchApplier:
    @staticmethod
    def _evolve_regions(src: str) -> List[Tuple[int, int]]:
        return [(m.start(1), m.end(1)) for m in _EVOLVE_RE.finditer(src)]

    @staticmethod
    def apply_diff(source: str, diff_text: str) -> Optional[str]:
        regions = PatchApplier._evolve_regions(source)
        if not regions:
            return None  # nothing is editable

        new = source
        for search, replace in _DIFF_RE.findall(diff_text):
            pos = new.find(search)
            if pos == -1:
                return None  # SEARCH chunk not found verbatim

            # verify that the match lies wholly inside one evolve region
            in_block = any(start <= pos < end for start, end in regions)
            if not in_block:
                return None  # attempted edit outside allowed span

            new = new.replace(search, replace, 1)

        return None if new == source else new

    @staticmethod
    def is_valid(py_code: str) -> bool:
        try:
            compile(py_code, "<candidate>", "exec")
            return True
        except SyntaxError:
            return False