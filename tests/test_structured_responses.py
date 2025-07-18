import pytest
from alphaevolve.response_parser import parse_structured_response
from alphaevolve.patcher import PatchApplier

def test_parse_structured_response():
    """Test parsing structured LLM responses."""
    
    # Test with both explanation and code
    response = """### Explanation
This is a detailed explanation of the changes I'm making to improve the algorithm.

### Code
<<<<<<< SEARCH
def old_function():
    return 1
=======
def new_function():
    return 2
>>>>>>> REPLACE"""
    
    explanation, code = parse_structured_response(response)
    assert explanation == "This is a detailed explanation of the changes I'm making to improve the algorithm."
    assert code is not None and "def new_function():" in code
    
    # Test with only code (should fail when explanation required)
    response = """<<<<<<< SEARCH
def old_function():
    return 1
=======
def new_function():
    return 2
>>>>>>> REPLACE"""
    
    explanation, code = parse_structured_response(response)
    assert explanation is None
    assert code is not None and "def new_function():" in code
    
    # Test full file replacement
    response = """### Explanation
This requires a complete rewrite of the file.

### Code
```python
### Full File Replacement
def completely_new_function():
    return "new implementation"
```"""
    
    explanation, code = parse_structured_response(response)
    assert explanation == "This requires a complete rewrite of the file."
    assert code is not None and "def completely_new_function():" in code

if __name__ == "__main__":
    pytest.main([__file__])