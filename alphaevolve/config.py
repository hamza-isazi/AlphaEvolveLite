from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class ExpCfg:
    label: str
    notes: str


@dataclass
class LLMCfg:
    provider: str
    model: str


@dataclass
class EvolCfg:
    population_size: int
    temperature: float
    max_generations: int
    inspiration_count: int


@dataclass
class Config:
    db_uri: str
    exp: ExpCfg
    llm: LLMCfg
    evolution: EvolCfg
    problem_entry: str
    problem_eval: str

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
        )
