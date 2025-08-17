import pytest
import time
import tempfile
import os
import logging
from unittest.mock import patch
from alphaevolve.llm import LLMEngine
from alphaevolve.llm import LLMCfg, ModelCfg
from alphaevolve.problem import Problem


def test_llm_generate_timeout():
    """Test that LLM generate function respects timeout settings."""
    
    # Create a minimal LLM config with a very short timeout
    model_cfg = ModelCfg(
        name="gemini-2.5-flash-lite",  # Use a real model for testing
        probability=1.0,
        llm_timeout=0.1  # Very short timeout - 100ms
    )
    
    llm_cfg = LLMCfg(
        provider="gemini",
        models=[model_cfg],
        system_prompt="You are a helpful assistant."
    )
    
    # Create a real LLM client
    # Create logger
    logger = logging.getLogger("test_llm_timeout")
    logger.setLevel(logging.INFO)
    
    # Create LLM engine
    llm_engine = LLMEngine(llm_cfg, logger)
    
    # Measure the time it takes to generate a response
    start_time = time.time()
    # Test that a timeout occurs with a very short timeout
    with pytest.raises(Exception) as exc_info:
        # This should timeout due to the very short timeout setting
        llm_engine.generate("Please provide a very detailed and lengthy response that will take a long time to generate.")
    print("LLM Engine Exception raised:", exc_info.value)
    print("Time taken:", time.time() - start_time)
    # The exception should be related to timeout (could be various types depending on the client)
        # For requests, timeout exceptions are typically requests.exceptions.Timeout
    assert ("timeout" in str(exc_info.value).lower() or 
            "timed out" in str(exc_info.value).lower() or
            "timeout" in str(type(exc_info.value).__name__).lower())
    assert "timeout" in str(exc_info.value).lower() or "timed out" in str(exc_info.value).lower()


def test_evaluate_with_timeout():
    """Test that evaluate_with_timeout function respects timeout settings."""
    
    # Create a temporary Python file that will run for a long time
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""
import time
import sys

# This will run for 10 seconds
time.sleep(10)
print("This should not be reached due to timeout")
sys.exit(0)
""")
        temp_file_path = f.name
    
    try:
        # Create a simple evaluator that just returns a score
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as eval_f:
            eval_f.write("""
def evaluate(path):
    # Import and run the program at the given path
    import subprocess
    import sys
    
    try:
        result = subprocess.run([sys.executable, path], 
                              capture_output=True, 
                              text=True, 
                              timeout=15)  # This timeout is longer than our test timeout
        if result.returncode == 0:
            return 1.0  # Success
        else:
            return 0.0  # Failure
    except subprocess.TimeoutExpired:
        return 0.0  # Timeout
    except Exception:
        return 0.0  # Any other error
""")
            evaluator_path = eval_f.name
        
        try:
            # Create problem instance
            problem = Problem("dummy_entry.py", evaluator_path)
            
            start_time = time.time()
            # Test with a very short timeout
            with pytest.raises(TimeoutError) as exc_info:
                problem.evaluate_with_timeout(temp_file_path, 0.1)  # 100ms timeout
            print("Evaluate Engine Exception raised:", exc_info.value)
            print("Time taken:", time.time() - start_time)
            # Verify the timeout error message
            assert "timed out after 0.1 seconds" in str(exc_info.value)
            
        finally:
            # Clean up evaluator file
            os.unlink(evaluator_path)
    
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)


if __name__ == "__main__":
    pytest.main([__file__]) 