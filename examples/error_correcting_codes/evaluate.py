import importlib.util
import sys
import random
import string

def import_solution(candidate_path):
    spec = importlib.util.spec_from_file_location("solution", candidate_path)
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)
    return solution

def introduce_error(s, max_errors=1):
    if max_errors == 0 or len(s) == 0:
        return s
    idx = random.randint(0, len(s) - 1)
    # Change to a random different character
    new_char = random.choice([c for c in string.printable if c != s[idx]])
    return s[:idx] + new_char + s[idx+1:]

def evaluate(candidate_path):
    solution = import_solution(candidate_path)
    test_cases = [
        "hello world",
        "error correcting codes",
        "1234567890",
        "test",
        "a",
        "",
        "The quick brown fox jumps over the lazy dog."
    ]
    total = 0
    correct = 0
    for msg in test_cases:
        encoded = solution.encode(msg)
        # Test no error
        decoded = solution.decode(encoded)
        total += 1
        if decoded == msg:
            correct += 1
        
        # Test with 1 error (if encoded is not empty)
        if encoded:
            corrupted = introduce_error(encoded, 1)
            decoded = solution.decode(corrupted)
            total += 1
            if decoded == msg:
                correct += 1
    score = correct / total if total > 0 else 0.0
    print(score)

if __name__ == "__main__":
    # evaluate("examples/error_correcting_codes/initial_program.py")
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <program_path>")
        sys.exit(1)
    
    program_path = sys.argv[1]
    
    result = evaluate(program_path, imports_allowed=True)
    print(result)