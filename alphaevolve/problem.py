import importlib.util
from pathlib import Path
from typing import Callable


class Problem:
    def __init__(self, entry_script: str, evaluator_path: str):
        self.entry_path = Path(entry_script)
        self.evaluate: Callable[[str], float] = self._load_eval(evaluator_path)

    @staticmethod
    def _load_eval(path: str) -> Callable[[str], float]:
        spec = importlib.util.spec_from_file_location("eval_mod", path)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        assert spec and spec.loader
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        if not hasattr(module, "evaluate"):
            raise AttributeError("Evaluator must expose `evaluate(path) -> float`.")
        return getattr(module, "evaluate")
