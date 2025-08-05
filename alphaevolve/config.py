from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import List, Optional

@dataclass
class ExpCfg:
    label: str
    notes: str
    save_top_k: int


@dataclass
class ModelCfg:
    """Configuration for a single model."""
    name: str
    probability: float
    temperature: Optional[float] = None


@dataclass
class LLMCfg:
    provider: str
    models: List[ModelCfg]
    system_prompt: str = "You are an expert software developer evolving Python code using diffs."
    retry_model: Optional[str] = None  # Model name to use for retries and feedback (if None, uses same model selection logic)
    llm_timeout: float = 120.0  # Global timeout in seconds for LLM API calls


@dataclass
class EvolCfg:
    population_size: int
    temperature: float
    max_generations: int
    inspiration_count: int
    max_retries: int = 3   # Number of retries for failed program generation
    eval_timeout: float = 60.0  # Timeout in seconds for evaluation runs
    enable_feedback: bool = True  # Enable LLM-generated feedback for successful programs
    recent_generations: int = 5  # Number of recent generations to consider for inspiration selection
    recent_percentile: float = 10.0  # Percentile threshold for recent generation selection (0-100)
    selection_method: str = "enhanced_inspiration"  # Method for inspiration selection: "boltzmann", "top_k_and_random", or "enhanced_inspiration"
    tabu_search_probability: float = 0.5  # Probability of using tabu search (fundamentally new approach) vs improvement


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
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        llm_data = data["llm"]
        models = [ModelCfg(**model_data) for model_data in llm_data["models"]]
        # Check that the sum of probabilities is 1.0
        total_probability = sum(model.probability for model in models)  
        if total_probability != 1.0:
            raise ValueError(f"The sum of probabilities for the models must be 1.0, but is {total_probability}")

        llm_cfg = LLMCfg(
            provider=llm_data["provider"],
            models=models,
            system_prompt=llm_data.get("system_prompt", "You are an expert software developer evolving Python code using diffs."),
            retry_model=llm_data.get("retry_model"),
            llm_timeout=llm_data.get("llm_timeout", 300.0)
        )
        
        return cls(
            db_uri=data["db_uri"],
            exp=ExpCfg(**data["experiment"]),
            llm=llm_cfg,
            evolution=EvolCfg(**data["evolution"]),
            problem_entry=data["problem"]["entry_script"],
            problem_eval=data["problem"]["evaluator"],
            debug=data.get("debug", False),  # Default to False if not specified
        )