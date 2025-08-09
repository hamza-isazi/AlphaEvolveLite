from dataclasses import dataclass
from typing import List, Optional
from openai import NOT_GIVEN


@dataclass
class ModelCfg:
    """Configuration for a single model."""
    name: str
    probability: float
    temperature: Optional[float] = NOT_GIVEN
    reasoning_effort: Optional[str] = NOT_GIVEN
    retry_model: Optional[str] = None  # Model name to use for retries and feedback for this specific model (if None, reuses the same model)
    provider: Optional[str] = None  # Provider for this specific model (if None, uses global provider)

    def __post_init__(self):
        if self.reasoning_effort is not NOT_GIVEN and self.reasoning_effort is not None:
            valid_efforts = {"low", "medium", "high"}
            if self.reasoning_effort not in valid_efforts:
                raise ValueError(f"Model {self.name}: Invalid reasoning_effort '{self.reasoning_effort}'. Must be one of {valid_efforts}.")


@dataclass
class LLMCfg:
    provider: str
    models: List[ModelCfg]
    system_prompt: str = "You are an expert software developer evolving Python code using diffs."
    llm_timeout: float = 120.0  # Global timeout in seconds for LLM API calls

    def __post_init__(self):
        # Set default providers for models that don't have one
        for model in self.models:
            if model.provider is None:
                model.provider = self.provider
        
        # Validate that the sum of probabilities is 1.0
        total_probability = sum(model.probability for model in self.models)  
        if total_probability != 1.0:
            raise ValueError(f"The sum of probabilities for the models must be 1.0, but is {total_probability}")


        # Validate that model names are unique
        names = [m.name for m in self.models]
        if len(names) != len(set(names)):
            raise ValueError("Model names must be unique.")

        # Validate that retry_model references exist
        model_names = {model.name for model in self.models}
        for model in self.models:
            if model.retry_model and model.retry_model not in model_names:
                raise ValueError(f"Model {model.name}: retry_model '{model.retry_model}' not found in available models")