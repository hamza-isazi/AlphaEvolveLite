import os
import time
import json
import random
import logging
from typing import List, cast
from openai import NOT_GIVEN, OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .config import LLMCfg, ModelCfg
from .utils import timeout

class LLMEngine:
    """LLM engine with conversation management and internal metric tracking."""
    
    def __init__(self, llm_cfg: LLMCfg, client: OpenAI, logger: logging.Logger) -> None:
        self.llm_cfg = llm_cfg
        self.system_prompt = llm_cfg.system_prompt
        self.client = client
        self.selected_model: ModelCfg = None
        self.logger = logger
        # Internal metric tracking
        self._total_llm_time = 0.0
        self._total_tokens = 0
        
        # Select initial model
        self._select_model()
        
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
        """Select a model specifically for retries and feedback, also removes temperature if specified.
        If retry_model is specified in config, use that model."""
        
        if self.llm_cfg.retry_model:
            for model in self.llm_cfg.models:
                if model.name == self.llm_cfg.retry_model:
                    self.selected_model = model
                    self.selected_model.temperature = None
                    return
            self.logger.warning(f"Retry model '{self.llm_cfg.retry_model}' not found in available models, keeping current model")
        else:
            self.selected_model.temperature = None

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
        # Make the API call
        response = self.client.chat.completions.create(
            model=self.selected_model.name,
            messages=self.messages,
            temperature=self.selected_model.temperature if self.selected_model.temperature else NOT_GIVEN
        )
        
        content = response.choices[0].message.content
        
        # Extract token usage information
        usage = response.usage
        total_tokens = usage.total_tokens if usage else 0
        
        return content.strip() if content else "", total_tokens

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """Generate a response from the LLM with timeout handling and retry logic."""
        # Track response time
        start_time = time.time()
        # Add the user prompt to the conversation
        self.add_message("user", prompt)
        
        last_exception = None
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
                last_exception = e
                if attempt < max_retries - 1:
                    self.logger.warning(f"Attempt {attempt + 1}/{max_retries}: Error getting response from provider {self.llm_cfg.provider} and model {self.selected_model.name}: {str(e)}")
                else:
                    self.logger.error(f"All {max_retries} attempts failed. Last error: {str(e)}")
                    raise last_exception
    
    def get_used_model(self) -> str:
        """Get the name of the currently selected model."""
        return self.selected_model.name


def create_llm_client(llm_cfg: LLMCfg) -> OpenAI:
    """Create the appropriate OpenAI client based on provider."""
    if llm_cfg.provider.lower() == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in your environment.")
        return OpenAI(api_key=api_key)
    elif llm_cfg.provider.lower() == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY in your environment.")
        return OpenAI(
            api_key=api_key, 
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    elif llm_cfg.provider.lower() == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY in your environment.")
        return OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_cfg.provider}")