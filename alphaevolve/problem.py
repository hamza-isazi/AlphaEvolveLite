import importlib.util
import time
from pathlib import Path
from typing import Callable, Tuple, Optional
import io
import sys

from .utils import timeout as timeout_decorator

class Problem:
    def __init__(self, entry_script: str, evaluator_path: str):
        self.entry_path = Path(entry_script)
        self.evaluate: Callable[[str], float] = self._load_eval(evaluator_path)

    @staticmethod
    def _load_eval(path: str) -> Callable[[str], float]:
        spec = importlib.util.spec_from_file_location("eval_mod", path)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        assert spec and spec.loader
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        if not hasattr(module, "evaluate"):
            raise AttributeError("Evaluator must expose `evaluate(path) -> float`.")
        return getattr(module, "evaluate")

    def _evaluate_with_logs(self, path: str) -> Tuple[float, str]:
        """
        Wrapper around the evaluate function that captures logs. DO NOT MAKE ANY STATE CHANGES HERE,
        they will not persist since this function is called in a subprocess by the timeout decorator.
        
        Returns:
            Tuple of (score, logs) where logs is the captured stdout/stderr
        """
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        # Create string buffers to capture output
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            # Redirect stdout and stderr to our buffers
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer
            
            # Call the original evaluate function
            score = self.evaluate(path)
            
            # Capture the logs
            stdout_logs = stdout_buffer.getvalue()
            stderr_logs = stderr_buffer.getvalue()
            
            # Combine logs
            logs = f"STDOUT:\n{stdout_logs}\nSTDERR:\n{stderr_logs}"
            
            return score, logs
            
        finally:
            # Restore original stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            stdout_buffer.close()
            stderr_buffer.close()

    def evaluate_with_timeout(self, path: str, timeout: float) -> Tuple[float, float, Optional[str]]:
        """
        Run the evaluate function with a timeout using the timeout decorator.

        Args:
            path: The path of the program to evaluate
            timeout: Timeout in seconds

        Returns:
            Tuple of (score, execution_time, logs) where execution_time is in seconds and logs is the evaluation output

        Raises:
            TimeoutError: If the evaluation exceeds the specified timeout
            Exception: Any exception that occurs during evaluation
        """
        try:
            # Use the timeout decorator to wrap the evaluation
            evaluate_with_timeout = timeout_decorator(
                timeout, 
                f"Evaluation timed out after {timeout} seconds"
            )(self._evaluate_with_logs)
            
            start_time = time.time()
            score, logs = evaluate_with_timeout(path)
            execution_time = time.time() - start_time
            
            return (score, execution_time, logs)
            
        except Exception as e:
            # Re-raise the exception to preserve the original behavior
            raise e