import importlib.util
import os
from pathlib import Path
import time
import sys
import random

# Directory for optional test files (if you want to use files)
script_dir = Path(__file__).parent.resolve()
filename = "Combinations_of_ascending_and_descending_two_sub_arrays-input-1000"
INPUTS_DIR = script_dir / "inputs" / (filename + ".txt")

def _check_for_imports(program_path: str):
    """Check for forbidden imports (e.g., built-in sorting functions)."""
    forbidden_libs = ['sorted', 'sort', 'heapq', 'bisect']
    with open(program_path, 'r') as f:
        content = f.read()
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        line_lower = line.lower().strip()
        if "import" in line_lower or "from" in line_lower:
            for lib in forbidden_libs:
                if f"import {lib}" in line_lower or f"from {lib}" in line_lower:
                    raise ValueError(f"Forbidden library import found on line {i}: '{line.strip()}'. Built-in sorting functions are not allowed in this experiment.")

def _load_module(path: str, imports_allowed: bool = False):
    if not imports_allowed:
        _check_for_imports(path)
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

def _create_mismatch_error_message(original, sorted_result, expected):
    msg = "Sorting failed:\n"
    msg += f"  Original:   {repr(original)}\n"
    msg += f"  Result:     {repr(sorted_result)}\n"
    msg += f"  Expected:   {repr(expected)}\n"
    # Find first difference
    min_len = min(len(sorted_result), len(expected))
    for i in range(min_len):
        if sorted_result[i] != expected[i]:
            msg += f"  First difference at position {i}\n"
            break
    if len(sorted_result) != len(expected):
        msg += f"  Length mismatch: result {len(sorted_result)}, expected {len(expected)}\n"
    return msg

def load_test_cases():
    # Optionally load from files in INPUTS_DIR, or just use hardcoded cases
    if INPUTS_DIR.exists():
        cases = []
        for fname in os.listdir(INPUTS_DIR):
            if fname.endswith('.txt'):
                with open(INPUTS_DIR / fname, 'r') as f:
                    # Read numbers separated by newlines
                    numbers = []
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            try:
                                # Try to convert to float first, then int if possible
                                num = float(line)
                                if num.is_integer():
                                    numbers.append(int(num))
                                else:
                                    numbers.append(num)
                            except ValueError:
                                # If not a number, skip this line
                                continue
                    if numbers:  # Only add non-empty arrays
                        cases.append(numbers)
        return cases if cases else [
            [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],  # Fallback test case
        ]
    else:
        return [
            [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],  # Integers with duplicates
            [5.5, 2.1, 8.9, 1.2, 3.7],  # Floats
            ["banana", "apple", "cherry", "date"],  # Strings
            [1, "hello", 3.14, True],  # Mixed types (should fail gracefully)
            [],  # Empty array
            [42],  # Single element
            [9, 8, 7, 6, 5, 4, 3, 2, 1],  # Reverse sorted
            [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Already sorted
            [1, 1, 1, 1, 1],  # All same elements
            list(range(100)),  # Large array
            list(range(100, 0, -1)),  # Large reverse sorted
            [random.randint(1, 1000) for _ in range(50)],  # Random integers
        ]

def evaluate(program_path: str, imports_allowed: bool = False) -> float:
    mod = _load_module(program_path, imports_allowed)
    sort_func = getattr(mod, "sort")
    test_cases = load_test_cases()
    total = 0
    correct = 0
    total_time = 0.0
    all_elapsed_times = []
    
    for test_array in test_cases:
        try:
            # Create a copy to avoid modifying the original
            input_array = test_array.copy()
            
            start = time.perf_counter()
            result = sort_func(input_array)
            elapsed = time.perf_counter() - start
            all_elapsed_times.append(elapsed)
            total_time += elapsed
            total += 1
            
            # Check if the result is correctly sorted
            try:
                expected = sorted(test_array)
                if result == expected:
                    correct += 1
                else:
                    print(_create_mismatch_error_message(test_array, result, expected))
            except TypeError:
                # For mixed-type arrays that can't be sorted, just check if result matches input
                if result == test_array:
                    correct += 1
                else:
                    print(f"Mixed-type array handling failed: {test_array} -> {result}")
                    
        except Exception as e:
            print(f"Error processing test case {test_array}: {e}")
            continue
    
    correctness = correct / total if total > 0 else 0.0
    avg_time = total_time / total if total > 0 else 0.0
    max_time = 0.01
    speed_score = max(0.0, 1.0 - avg_time / max_time)
    score = 0.7 * correctness + 0.3 * speed_score
    print(score)
    # Optionally, print timing info
    # print(f"Correctness: {correctness:.4f}, Avg time: {avg_time:.6f}s, Speed score: {speed_score:.4f}")
    return score

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <program_path>")
        sys.exit(1)
    program_path = sys.argv[1]
    result = evaluate(program_path, imports_allowed=True)
    print(result) 