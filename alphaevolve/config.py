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


class ConfigContext:
    """Context object that provides centralized access to configuration and common dependencies."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._database = None
        self._problem = None
        self._prompt_sampler = None
        self._llm = None
        self._patcher = None
    
    @property
    def database(self):
        """Lazy initialization of database connection."""
        if self._database is None:
            from .db import EvolutionaryDatabase
            self._database = EvolutionaryDatabase(self.cfg)
        return self._database
    
    @property
    def problem(self):
        """Lazy initialization of problem evaluator."""
        if self._problem is None:
            from .problem import Problem
            self._problem = Problem(self.cfg.problem_entry, self.cfg.problem_eval)
        return self._problem
    
    @property
    def prompt_sampler(self):
        """Lazy initialization of prompt sampler."""
        if self._prompt_sampler is None:
            from .prompts import PromptSampler
            self._prompt_sampler = PromptSampler(self.database)
        return self._prompt_sampler
    
    @property
    def llm(self):
        """Lazy initialization of LLM engine."""
        if self._llm is None:
            from .llm import create_llm_engine
            self._llm = create_llm_engine(self.cfg.llm)
        return self._llm
    
    @property
    def patcher(self):
        """Lazy initialization of patch applier."""
        if self._patcher is None:
            from .patcher import PatchApplier
            self._patcher = PatchApplier()
        return self._patcher
