import importlib.util
import os
from pathlib import Path
import time
import sys
import random
import string

# Directory for optional test files (if you want to use files)
script_dir = Path(__file__).parent.resolve()
INPUTS_DIR = script_dir / "inputs"

def _check_for_imports(program_path: str):
    """Check for forbidden imports (e.g., third-party ECC libraries)."""
    forbidden_libs = ['reedsolo', 'bchlib', 'pyfinite', 'numpy', 'scipy']
    with open(program_path, 'r') as f:
        content = f.read()
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        line_lower = line.lower().strip()
        if "import" in line_lower:
            for lib in forbidden_libs:
                if f"import {lib}" in line_lower or f"from {lib}" in line_lower:
                    raise ValueError(f"Forbidden library import found on line {i}: '{line.strip()}'. Not allowed in this experiment.")

def _load_module(path: str, imports_allowed: bool = False):
    if not imports_allowed:
        _check_for_imports(path)
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

def introduce_error(s, max_errors=1):
    """Introduce up to max_errors single-character errors in the string s."""
    s = list(s)
    for _ in range(max_errors):
        if not s:
            break
        idx = random.randint(0, len(s) - 1)
        new_char = random.choice([c for c in string.printable if c != s[idx]])
        s[idx] = new_char
    return ''.join(s)

def _create_mismatch_error_message(original, encoded, corrupted, decoded):
    msg = "Decoding failed:\n"
    msg += f"  Original:   {repr(original)}\n"
    msg += f"  Encoded:    {repr(encoded)}\n"
    msg += f"  Corrupted:  {repr(corrupted)}\n"
    msg += f"  Decoded:    {repr(decoded)}\n"
    # Find first difference
    min_len = min(len(original), len(decoded))
    for i in range(min_len):
        if original[i] != decoded[i]:
            msg += f"  First difference at position {i}\n"
            break
    if len(original) != len(decoded):
        msg += f"  Length mismatch: original {len(original)}, decoded {len(decoded)}\n"
    return msg

def load_test_cases():
    # Optionally load from files in INPUTS_DIR, or just use hardcoded cases
    if INPUTS_DIR.exists():
        cases = []
        for fname in os.listdir(INPUTS_DIR):
            with open(INPUTS_DIR / fname, 'r') as f:
                cases.append(f.read().strip())
        return cases
    else:
        return [
            "hello world",
            "error correcting codes",
            "1234567890",
            "test",
            "a",
            "",
            "The quick brown fox jumps over the lazy dog."
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        ]

def evaluate(program_path: str, imports_allowed: bool = False) -> float:
    mod = _load_module(program_path, imports_allowed)
    encode = getattr(mod, "encode")
    decode = getattr(mod, "decode")
    test_cases = load_test_cases()
    total = 0
    correct = 0
    total_time = 0.0
    for msg in test_cases:
        # No error test
        start = time.perf_counter()
        encoded = encode(msg)
        decoded = decode(encoded)
        elapsed = time.perf_counter() - start
        total_time += elapsed
        total += 1
        if decoded == msg:
            correct += 1
        else:
            print(_create_mismatch_error_message(msg, encoded, encoded, decoded))
        # Single error test (if encoded is not empty)
        if encoded:
            corrupted = introduce_error(encoded, 1)
            start = time.perf_counter()
            decoded = decode(corrupted)
            elapsed = time.perf_counter() - start
            total_time += elapsed
            total += 1
            if decoded == msg:
                correct += 1
            else:
                print(_create_mismatch_error_message(msg, encoded, corrupted, decoded))
    score = correct / total if total > 0 else 0.0
    print(score)
    # Optionally, print timing info
    # print(f"Total time: {total_time:.4f} seconds for {total} tests")
    return score

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <program_path>")
        sys.exit(1)
    program_path = sys.argv[1]
    result = evaluate(program_path, imports_allowed=True)
    print(result)