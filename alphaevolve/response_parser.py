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

def parse_structured_response(response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a structured LLM response to extract explanation and code sections.
    
    Args:
        response: The full LLM response text
        
    Returns:
        Tuple of (explanation, code) where both can be None if not found
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
    
    # If no structured format found, treat entire response as code
    if explanation is None and code is None:
        code = response
    
    return explanation, code 