import importlib.util
import tracemalloc
import time
import random
import sys

EXPECTED_OPS = 100_000
CACHE_CAPACITY = 10_000

def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod

def evaluate(program_path: str) -> dict:
    mod = _load_module(program_path)
    LRUCache = getattr(mod, "LRUCache")

    ops = EXPECTED_OPS
    capacity = CACHE_CAPACITY
    keys = [random.randint(0, capacity * 2) for _ in range(ops)]

    cache = LRUCache(capacity)

    tracemalloc.start()
    start = time.perf_counter()

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