"""
Hashcode_2020_qualification_round Eval Script for Book Scanning Problem
Adapted from https://github.com/pgimalac/hashcode-evaluator/blob/master/2020-qualification-round/eval.py
Best recorded competition score: 27203691
"""

import sys
import time
from pathlib import Path

def evaluate_input_output(input_text: str, output_text: str) -> int:
    def error(msg):
        raise ValueError(msg)

    input_lines = input_text.strip().splitlines()
    output_lines = output_text.strip().splitlines()

    def next_input():
        return input_lines.pop(0)

    def next_output():
        return output_lines.pop(0)

    def readints(line):
        return list(map(int, line.strip().split()))

    B, L, D = readints(next_input())
    scores = readints(next_input())

    libraries = []
    for _ in range(L):
        Nj, Tj, Mj = readints(next_input())
        ids = set(readints(next_input()))
        libraries.append((Nj, Tj, Mj, ids))

    score = 0
    books = set()
    time = 0

    A = int(next_output())
    if A < 0 or A > L:
        error(f"The number of libraries must be between 0 and {L}")

    for i in range(A):
        Y, K = readints(next_output())
        if Y < 0 or Y >= L:
            error(f"Line {2*i+2}: library id {Y} out of range")
        Nj, Tj, Mj, idset = libraries[Y]
        if K < 1 or K > Nj:
            error(f"Line {2*i+2}: scanned book count {K} invalid for library {Y}")
        ids = readints(next_output())
        if len(ids) != K or len(ids) != len(set(ids)):
            error(f"Line {2*i+3}: scanned book ids must be unique and match K")
        if any(e not in idset for e in ids):
            error(f"Line {2*i+3}: some books not in library {Y}")

        time += Tj
        n = min(K, (D - time) * Mj)
        for e in ids[:n]:
            if e not in books:
                books.add(e)
                score += scores[e]

    if output_lines:
        error("Too many lines in output")

    return score


def evaluate(program_path: str) -> int:
    script_dir = Path(__file__).parent.resolve()
    inputs_path = script_dir / "inputs"
    print(inputs_path)

    # Import the run function from the program
    import importlib.util
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None:
        raise ImportError(f"Could not load module from {program_path}")
    program_module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not load module from {program_path}")
    spec.loader.exec_module(program_module)
    run_function = program_module.run

    results = {}
    total_time = 0.0

    for file in sorted(Path(inputs_path).glob("*.in")):
        with open(file) as f:
            input_text = f.read()
        try:
            start = time.time()
            output_text = run_function(input_text)
            elapsed = time.time() - start
            total_time += elapsed

            score = evaluate_input_output(input_text, output_text)
            results[file.name] = score
        except Exception as e:
            results[file.name] = f"Evaluation error: {str(e)}"

    results['combined_score'] = sum(score for score in results.values() if isinstance(score, int))
    results['throughput'] = round(1/total_time, 3) if total_time != 0 else 0
    return results['combined_score']

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <program_path>")
        sys.exit(1)
    result = evaluate(sys.argv[1])
    print(result)
