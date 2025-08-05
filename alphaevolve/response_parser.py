import re
from typing import Tuple, Optional

_EXPLANATION_RE = re.compile(
    r"###\s*Explanation\s*\n(.*?)(?=\n###\s*Code\s*\n)",
    re.DOTALL,
)
_CODE_RE = re.compile(
    r"###\s*Code\s*\n(.*)",
    re.DOTALL,
)


def parse_code_response(response: str) -> Tuple[Optional[str], str]:
    """
    Parse a structured LLM response to extract code with an optional explanation.
    
    Args:
        response: The full LLM response text
        
    Returns:
        Tuple of (explanation, code) where explanation can be None if not found and code is the entire response if no code section is found
    """
    # Extract explanation section
    explanation_match = _EXPLANATION_RE.search(response)
    explanation = None
    if explanation_match:
        explanation = explanation_match.group(1).strip()
    
    # Extract code section
    code_match = _CODE_RE.search(response)
    code = None
    if code_match:
        code = code_match.group(1).strip()
    else:
        # If no code section is found, return the entire response as code
        code = response
    return explanation, code 