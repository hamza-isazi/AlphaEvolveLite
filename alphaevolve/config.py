from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class ExpCfg:
    label: str
    notes: str
    save_top_k: int


@dataclass
class LLMCfg:
    provider: str
    model: str
    temperature: float = 0.7
    system_prompt: str = "You are an expert software developer evolving Python code using diffs."


@dataclass
class EvolCfg:
    population_size: int
    temperature: float
    max_generations: int
    inspiration_count: int
    max_retries: int = 3   # Number of retries for failed program generation
    evaluation_timeout: float = 60.0  # Timeout in seconds for evaluation runs


@dataclass
class Config:
    db_uri: str
    exp: ExpCfg
    llm: LLMCfg
    evolution: EvolCfg
    problem_entry: str
    problem_eval: str
    debug: bool = False  # Control logging verbosity

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            db_uri=data["db_uri"],
            exp=ExpCfg(**data["experiment"]),
            llm=LLMCfg(**data["llm"]),
            evolution=EvolCfg(**data["evolution"]),
            problem_entry=data["problem"]["entry_script"],
            problem_eval=data["problem"]["evaluator"],
            debug=data.get("debug", False),  # Default to False if not specified
        )
