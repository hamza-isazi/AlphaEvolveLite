import importlib.util
import os
from pathlib import Path
import time
import sys

# Get the directory where this script is located
script_dir = Path(__file__).parent.resolve()
CORPUS_DIR = script_dir / "cantrbry"

def _check_for_imports(program_path: str):
    """Check if the program contains any import statements and raise an error if found."""
    with open(program_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        if "import" in line:
            raise ValueError(f"Import statement found on line {i}: '{line.strip()}'. No imports are allowed in this experiment.")

def _load_module(path: str, imports_allowed: bool = False):
    # First check for imports (no imports allowed)
    if not imports_allowed:
        _check_for_imports(path)
    
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod

# Load all files
def load_corpus():
    files = [f for f in os.listdir(CORPUS_DIR) if (CORPUS_DIR / f).is_file()]
    return files

def evaluate(program_path: str, imports_allowed: bool = False) -> float:
    mod = _load_module(program_path, imports_allowed)
    compress = getattr(mod, "compress")
    decompress = getattr(mod, "decompress")

    total_elapsed = 0
    total_compressed_size = 0
    files = load_corpus()

    for file in files:
        with open(CORPUS_DIR / file, "rb") as f:
            raw_bytes = f.read()
            text = raw_bytes.decode("latin1")

        start = time.perf_counter()
        compressed = compress(text)
        decompressed = decompress(compressed)
        elapsed = time.perf_counter() - start

        assert decompressed == text, f"Mismatch on file {file}"
        total_compressed_size += len(compressed)
        total_elapsed += elapsed
    score_dict = {
        "total_compressed_size": total_compressed_size,
        "total_elapsed": total_elapsed,
        "combined_score": 1.0 / (total_compressed_size / 1024**2) if total_compressed_size > 0 else 0.0,
    }

    return score_dict['combined_score']

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <program_path>")
        sys.exit(1)
    
    program_path = sys.argv[1]
    
    result = evaluate(program_path, imports_allowed=True)
    print(result)