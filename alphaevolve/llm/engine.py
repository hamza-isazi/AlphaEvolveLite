import time
import json
import random
import logging
from typing import List, cast
from openai import NOT_GIVEN
from openai.types.chat import ChatCompletionMessageParam

from .config import LLMCfg, ModelCfg
from .clients import global_client_pool
from ..utils import timeout


class LLMAPIError(Exception):
    """Exception raised when the LLM API returns an error."""
    pass


class LLMEngine:
    """LLM engine with conversation management and internal metric tracking."""
    
    def __init__(self, llm_cfg: LLMCfg, logger: logging.Logger) -> None:
        self.llm_cfg = llm_cfg
        self.system_prompt = llm_cfg.system_prompt
        self.selected_model: ModelCfg = None
        self.logger = logger
        # Internal metric tracking
        self._total_llm_time = 0.0
        self._total_tokens = 0
        
        # Select initial model
        self._select_model()

        # Init the client pool
        self._client_pool = global_client_pool()
        
        # Initialize conversation with system prompt
        self.messages: List[ChatCompletionMessageParam] = [
            cast(ChatCompletionMessageParam, {"role": "system", "content": self.system_prompt})
        ]

    def _select_model(self) -> None:
        """Select a model based on probabilities."""
        # Use random.choices with weights for simple weighted selection
        self.selected_model = random.choices(
            self.llm_cfg.models, 
            weights=[model.probability for model in self.llm_cfg.models], 
            k=1
        )[0]

    def select_retry_model(self) -> None:
        """Select a model specifically for retries and feedback, also unsets temperature if specified
        (temperature encourages creativity and tends to be less reliable for retries).
        If the current model has a retry_model specified, use that model. Otherwise, keep the current model."""
        
        if self.selected_model.retry_model:
            for model in self.llm_cfg.models:
                if model.name == self.selected_model.retry_model:
                    self.selected_model = model
                    self.selected_model.temperature = NOT_GIVEN
                    return
        else:
            self.selected_model.temperature = NOT_GIVEN

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        message = cast(ChatCompletionMessageParam, {"role": role, "content": content})
        self.messages.append(message)

    def reset_conversation(self) -> None:
        """Reset the conversation history, keeping only the system prompt."""
        self.messages = [cast(ChatCompletionMessageParam, {"role": "system", "content": self.system_prompt})]

    def get_conversation_json(self) -> str:
        """Get the full conversation history as a JSON string."""
        return json.dumps(self.messages, indent=2)

    def get_metrics(self) -> tuple[float, int]:
        """Get the total LLM time and tokens used."""
        return self._total_llm_time, self._total_tokens

    def reset_metrics(self) -> None:
        """Reset the internal metrics."""
        self._total_llm_time = 0.0
        self._total_tokens = 0

    def _generate_internal(self) -> str:
        """Internal method that performs the actual LLM generation. DO NOT MAKE ANY STATE CHANGES HERE,
        they will not persist since this function is called in a subprocess by the timeout decorator."""                
        # Get the appropriate client for the selected model
        client = self._client_pool.get(self.selected_model.provider)
        
        # Make the API call
        response = client.chat.completions.create(
            model=self.selected_model.name,
            messages=self.messages,
            temperature=self.selected_model.temperature,
            reasoning_effort=self.selected_model.reasoning_effort,
            timeout=self.llm_cfg.llm_timeout
        )
        
        content = response.choices[0].message.content
        
        # Extract token usage information
        usage = response.usage
        total_tokens = usage.total_tokens if usage else 0
        
        return content.strip() if content else "", total_tokens

    def generate(self, prompt: str, max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 30.0) -> str:
        """Generate a response from the LLM with timeout handling and retry logic with exponential backoff.
        Args:
            max_retries: Maximum number of retries
            base_delay: Base delay in seconds between retries
            max_delay: Maximum delay in seconds between retries
        Returns:
            The response from the LLM
        Raises:
            Exception: If all retries fail
        """
        # Track response time
        start_time = time.time()
        # Add the user prompt to the conversation
        self.add_message("user", prompt)
        
        # Try up to max_retries times
        for attempt in range(max_retries):
            try:
                # Use the timeout decorator to wrap the generation
                # (OpenAI's built-in timeout param does not work correctly)
                generate_with_timeout = timeout(
                    self.llm_cfg.llm_timeout, 
                    f"LLM generation timed out after {self.llm_cfg.llm_timeout} seconds"
                )(self._generate_internal)
                
                content, total_tokens = generate_with_timeout()
                
                # Check if we got a valid response (not empty or None)
                if not content or not content.strip():
                    raise ValueError(f"Empty response from {self.selected_model.name}")
                
                # Add the assistant's response to the conversation
                self.add_message("assistant", content)
                
                # Update internal metrics
                response_time = time.time() - start_time
                self._total_llm_time += response_time
                self._total_tokens += total_tokens
                
                return content
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    # Calculate exponential backoff delay
                    delay = min(base_delay * (2 ** (attempt + 1)) + random.uniform(0, 1), max_delay)
                    
                    self.logger.info(f"LLM Generation Attempt {attempt + 1}/{max_retries}: Error getting response from provider {self.selected_model.provider} and model {self.selected_model.name}: {e.__class__.__name__}: {e}")
                    self.logger.info(f"Retrying in {delay:.2f} seconds...")
                    
                    # Wait before retrying
                    time.sleep(delay)
                else:
                    self.logger.error(f"Attempt {attempt + 1}/{max_retries}: Error getting response from provider {self.selected_model.provider} and model {self.selected_model.name}: {e.__class__.__name__}: {e}")
                    raise LLMAPIError(f"Attempt {attempt + 1}/{max_retries}: Error getting response from provider {self.selected_model.provider} and model {self.selected_model.name}: {e.__class__.__name__}: {e}")
    
    def get_used_model(self) -> str:
        """Get the name of the currently selected model."""
        return self.selected_model.name