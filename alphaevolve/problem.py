import importlib.util
import multiprocessing as mp
from concurrent.futures import TimeoutError
from pathlib import Path
from typing import Callable
import traceback

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

    def evaluate_with_timeout(self, path: str, timeout: float) -> float:
        """
        Run the evaluate function with a timeout in a separate process.

        Args:
            path: The path of the program to evaluate
            timeout: Timeout in seconds

        Returns:
            The evaluation result (float)

        Raises:
            TimeoutError: If the evaluation exceeds the specified timeout
            Exception: Any exception that occurs during evaluation
        """
        # Define the target function that will run in a separate process
        def target(queue):
            try:
                score = self.evaluate(path)
                # Put the result in the queue for the parent process to retrieve
                queue.put(score)
            except Exception as e:
                # Format the exception with a simplified traceback (last frame)
                tb = traceback.extract_tb(e.__traceback__)
                if tb:
                    last_frame = tb[-1]
                    error_message = f"{str(e)}\n  File \"{last_frame.filename}\", line {last_frame.lineno}, in {last_frame.name}"
                    if last_frame.line:
                        error_message += f"\n    {last_frame.line.strip()}"
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
            
        # Return the evaluation score
        return result