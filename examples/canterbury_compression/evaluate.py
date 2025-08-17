import importlib.util
import os
from pathlib import Path
import time
import sys

# Get the directory where this script is located
script_dir = Path(__file__).parent.resolve()
CORPUS_DIR = script_dir / "cantrbry"

def _check_for_imports(program_path: str):
    """Check if the program contains any compression-related import statements and raise an error if found."""
    # List of compression-related libraries that are not allowed
    compression_libs = [
        'zlib', 'bz2', 'lzma', 'gzip', 'zipfile', 'tarfile', 'compress', 'decompress',
        'zstandard', 'lz4', 'snappy', 'brotli', 'lzo', 'quicklz', 'lzf'
    ]
    
    with open(program_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        line_lower = line.lower().strip()
        if "import" in line_lower:
            # Check for compression library imports
            for lib in compression_libs:
                if f"import {lib}" in line_lower or f"from {lib}" in line_lower:
                    raise ValueError(f"Compression library import found on line {i}: '{line.strip()}'. Compression libraries are not allowed in this experiment.")

def _load_module(path: str, imports_allowed: bool = False):
    # First check for imports (no imports allowed)
    if not imports_allowed:
        _check_for_imports(path)
    
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod

def _create_mismatch_error_message(file: str, original_text: str, compressed: bytes, decompressed: str) -> str:
    """Create a detailed error message for compression/decompression mismatches."""
    original_size = len(original_text)
    compressed_size = len(compressed)
    decompressed_size = len(decompressed)
    compression_ratio = compressed_size / original_size if original_size > 0 else 0
    
    # Find the first difference
    min_len = min(len(original_text), len(decompressed))
    first_diff_pos = -1
    for i in range(min_len):
        if original_text[i] != decompressed[i]:
            first_diff_pos = i
            break
    
    if first_diff_pos == -1 and len(original_text) != len(decompressed):
        first_diff_pos = min_len
    
    error_msg = f"Mismatch on file {file}:\n"
    error_msg += f"  Original size: {original_size} bytes\n"
    error_msg += f"  Compressed size: {compressed_size} bytes\n"
    error_msg += f"  Decompressed size: {decompressed_size} bytes\n"
    error_msg += f"  Compression ratio: {compression_ratio:.3f}\n"
    
    if first_diff_pos >= 0:
        error_msg += f"  First difference at position: {first_diff_pos}\n"
        
        # Check if it's a character mismatch      
        if first_diff_pos < min_len:
            # Show context around the character mismatch
            context_size = 50  # characters to show before and after
            start_pos = max(0, first_diff_pos - context_size)
            end_pos_orig = min(len(original_text), first_diff_pos + context_size + 1)
            end_pos_decomp = min(len(decompressed), first_diff_pos + context_size + 1)
            
            # Get context strings
            orig_context = original_text[start_pos:end_pos_orig]
            decomp_context = decompressed[start_pos:end_pos_decomp]
            
            error_msg += f"  Original:  ...{repr(orig_context)}...\n"
            error_msg += f"  Decompressed: ...{repr(decomp_context)}...\n"
            
            # Create arrow pointing to the mismatch in the Decompressed line
            arrow_pos = first_diff_pos - start_pos
            if arrow_pos >= 0:
                arrow_line = " " * (arrow_pos + 18) + "^ mismatch here\n"  # +18 for "Decompressed: ...'`text`'" prefix
                error_msg += f"  {arrow_line}"
        else:
            # Just a length difference
            if first_diff_pos >= len(original_text):
                error_msg += f"  Original text ended at position {len(original_text)}\n"
                error_msg += f"  Decompressed text has {len(decompressed)} characters (extra content)\n"
            else:
                error_msg += f"  Decompressed text ended at position {len(decompressed)}\n"
                error_msg += f"  Original text has {len(original_text)} characters (missing content)\n"
    
    return error_msg

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

        if decompressed != text:
            error_msg = _create_mismatch_error_message(file, text, compressed, decompressed)
            raise AssertionError(error_msg)
        total_compressed_size += len(compressed)
        total_elapsed += elapsed
    score_dict = {
        "total_compressed_size": total_compressed_size,
        "total_elapsed": total_elapsed,
        # Final score is the inverse of the compressed size in MB
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