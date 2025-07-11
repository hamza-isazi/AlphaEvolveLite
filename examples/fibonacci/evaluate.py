import importlib.util
import time

EXPECTED = [
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34,
    55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181,
]

def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("candidate", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def evaluate(path: str, num_iters=10) -> float:
    mod = _load_module(path)
    try:
        elapsed_times = []
        for i in range(num_iters):
            start = time.perf_counter()
            results = [mod.fib(i) for i in range(len(EXPECTED))]
            elapsed_times.append(time.perf_counter() - start)

        if results != EXPECTED:
            return 0.0
        else:
            # define a maximum time for calculating time penalty
            max_time = 0.005
            # penalty based on how close you are to max time
            avg_elapsed = sum(elapsed_times)/num_iters
            penalty = min(1, avg_elapsed / max_time)
            score = 1 - 0.5 * penalty

        return max(score, 0.0)

    except Exception:
        return 0.0
