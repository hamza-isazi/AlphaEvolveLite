import importlib.util
import time

EXPECTED = [
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34,
    55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181,
]

def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

def evaluate(path: str) -> float:
    mod = _load_module(path)
    try:
        start = time.perf_counter()
        results = [mod.fib(i) for i in range(len(EXPECTED))]
        elapsed = time.perf_counter() - start

        if results != EXPECTED:
            return 0.0
        
        # define a maximum time for scoring, e.g. 1 millisecond
        max_time = 0.01
        penalty = min(1.0, elapsed / max_time)
        score = 1.0 - penalty

        return max(score, 0.0)

    except Exception:
        return 0.0
