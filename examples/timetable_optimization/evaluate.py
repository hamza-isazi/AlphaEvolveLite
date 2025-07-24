#!/usr/bin/env python3
"""
Evaluation function for the timetable optimization problem.
This function takes a program path and returns a score based on how well
the generated timetable satisfies the optimization criteria.
"""

import importlib.util
import os
import sys
import tempfile
import subprocess
from pathlib import Path

# Get paths relative to this script
SCRIPT_DIR = Path(__file__).parent.resolve()
INPUTS_DIR = SCRIPT_DIR / "inputs"
OPT_TEST_PATH = SCRIPT_DIR / "opt_test.py"

def _load_module(path: str):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location("candidate", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def evaluate(path: str) -> float:
    """
    Evaluate a timetable generation program.
    
    Args:
        path: Path to the Python program that generates a timetable
        
    Returns:
        float: Score between 0.0 and 1.0, where 1.0 is perfect
    """
    # Load the candidate module
    mod = _load_module(path)
    
    # Create a temporary directory for the solution
    with tempfile.TemporaryDirectory() as temp_dir:
        # Change to a temp directory so we can clean up after ourselves and run the program
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:            
            # Run the main function from the module
            if hasattr(mod, 'main'):
                timetable = mod.main(INPUTS_DIR)
            else:
                # If no main function, try to call generate_timetable directly
                if hasattr(mod, 'generate_timetable'):
                    timetable = mod.generate_timetable(INPUTS_DIR)
                    mod.save_timetable(timetable, "solution.csv")
                else:
                    return 0.0
            
            # Check if solution.csv was created
            if not os.path.exists("solution.csv"):
                return 0.0
            
            # Run the evaluation using the original opt_test.py
            solution_path = os.path.join(temp_dir, "solution.csv")
            
            # Change back to original directory for opt_test.py
            os.chdir(original_cwd)
            
            result = subprocess.run([
                "python", 
                OPT_TEST_PATH, 
                solution_path
            ], capture_output=True, text=True, timeout=30)
            
            # Print the subprocess output so it gets captured by the log mechanism
            print("=== EVAL OUTPUT ===")
            print(result.stdout)
            print("=== END EVAL OUTPUT ===")
            
            # Parse the output to extract test results
            output = result.stdout
            
            # Count total tests and failures
            total_tests = 0
            passed_tests = 0
            
            for line in output.split('\n'):
                if "Total:" in line:
                    parts = line.split()
                    total_tests = int(parts[1])
                elif "Passed:" in line:
                    parts = line.split()
                    passed_tests = int(parts[1])
            
            if total_tests == 0:
                return 0.0
            
            # Calculate score as fraction of passed tests
            score = passed_tests / total_tests
            
            return max(0.0, min(1.0, score))
            
        finally:
            # Make sure we return to the original directory
            os.chdir(original_cwd)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <program_path>")
        sys.exit(1)
    
    score = evaluate(sys.argv[1])
    print(f"Score: {score}") 