import importlib.util
import tracemalloc
import time
import random
import sys

EXPECTED_OPS = 100_000
CACHE_CAPACITY = 10_000

def _check_for_imports(program_path: str):
    """Check if the program contains any import statements and raise an error if found."""
    with open(program_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        if "import" in line:
            raise ValueError(f"Import statement found on line {i}: '{line.strip()}'. No imports are allowed in this experiment.")

def _load_module(path: str):
    # First check for imports (no imports allowed)
    _check_for_imports(path)
    
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


def _run_unit_tests(LRUCache):
    cache = LRUCache(2)

    # Miss on empty cache
    assert cache.get(1) == -1, "Expected miss on empty cache"

    # Put and get
    cache.put(1, 100)
    assert cache.get(1) == 100, "Failed to get correct value after put"

    # Overwrite value
    cache.put(1, 101)
    assert cache.get(1) == 101, "Failed to update existing key"

    # Fill to capacity
    cache.put(2, 200)
    assert cache.get(2) == 200, "Expected value for key 2 after inserting it"
    assert cache.get(1) == 101, "Expected value for key 1 to remain unchanged after inserting key 2"

    # Trigger eviction (key 1 is LRU)
    cache.put(3, 300)
    assert cache.get(1) == 101, "Expected key 1 to remain as it was the most recently used"
    assert cache.get(2) == -1, "Expected the least recently used key (2) to be evicted"
    assert cache.get(3) == 300, "Expected key 3 to exist"

    # Use key 1, then evict key 3
    cache.get(1)
    cache.put(4, 400)
    assert cache.get(3) == -1, "Expected the least recently used key (3) to be evicted"
    assert cache.get(1) == 101, "Expected key 1 to remain as it was accessed most recently"
    assert cache.get(4) == 400, "Expected key 4 to exist"

def evaluate(program_path: str) -> dict:
    mod = _load_module(program_path)
    LRUCache = getattr(mod, "LRUCache")

    _run_unit_tests(LRUCache)

    ops = EXPECTED_OPS
    capacity = CACHE_CAPACITY
    keys = [random.randint(0, capacity * 2) for _ in range(ops)]

    tracemalloc.start()
    start = time.perf_counter()
    cache = LRUCache(capacity)

    for i, k in enumerate(keys):
        if i % 2 == 0:
            cache.put(k, k)
        else:
            _ = cache.get(k)

    runtime = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / 1e6
    score = 1.0 / (runtime * (1 + peak_mb))

    score_dict = {
        "latency_sec": runtime,
        "peak_memory_mb": peak_mb,
        "score": score,
    }

    return score_dict["score"]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <program_path>")
        sys.exit(1)
    
    program_path = sys.argv[1]
    
    try:
        result = evaluate(program_path)
        print(result)
    except Exception as e:
        print(f"Error: {e}")