import importlib.util
import multiprocessing as mp
import time
from concurrent.futures import TimeoutError
from pathlib import Path
from typing import Callable, Tuple, Optional
import traceback
import io
import sys

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
        Wrapper around the evaluate function that captures logs.
        
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
        Run the evaluate function with a timeout in a separate process.

        Args:
            path: The path of the program to evaluate
            timeout: Timeout in seconds

        Returns:
            Tuple of (score, execution_time, logs) where execution_time is in seconds and logs is the evaluation output

        Raises:
            TimeoutError: If the evaluation exceeds the specified timeout
            Exception: Any exception that occurs during evaluation
        """
        # Define the target function that will run in a separate process
        def target(queue):
            try:
                start_time = time.time()
                score, logs = self._evaluate_with_logs(path)
                execution_time = time.time() - start_time
                # Put the result, execution time, and logs in the queue
                queue.put((score, execution_time, logs))
            except Exception as e:
                # Format the exception with a traceback so the LLM can see where the error occurred
                tb = traceback.extract_tb(e.__traceback__)
                if tb:
                    # Skip the first frame (this function) and include all remaining frames
                    frames = tb[1:]
                    error_message = str(e)
                    for frame in frames:
                        error_message += f"\n  File \"{frame.filename}\", line {frame.lineno}, in {frame.name}"
                        if frame.line:
                            error_message += f"\n    {frame.line.strip()}"
                else:
                    error_message = str(e)
                # Put the formatted error message in the queue as an Exception
                queue.put(Exception(error_message))

        # Create a multiprocessing queue for inter-process communication
        queue = mp.Queue()
        
        # Create and start a new process to run the evaluation
        proc = mp.Process(target=target, args=(queue,))
        proc.start()
        
        # Wait for the process to complete or timeout
        proc.join(timeout)

        # Check if the process is still running (timed out)
        if proc.is_alive():
            # Force terminate the process
            proc.terminate()
            # Wait for termination to complete
            proc.join()
            # Clean up any remaining resources
            if proc.is_alive():
                proc.kill()
                proc.join()
            # Raise a TimeoutError to indicate timeout occurred
            raise TimeoutError(f"Evaluation timed out after {timeout} seconds")

        # Get the result from the queue
        # Note: queue.get() will block if the queue is empty, but since we've
        # confirmed the process has finished, this should not block
        result = queue.get()
        
        # If the result is an exception, re-raise it in the parent process
        if isinstance(result, Exception):
            # Re-raise the exception with context to preserve the original traceback
            raise result
            
        # Unpack the result (score, execution_time, logs)
        score, execution_time, logs = result
        
        # Return the evaluation score, execution time, and logs
        return (score, execution_time, logs)