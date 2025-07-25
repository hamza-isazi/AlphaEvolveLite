import os
import time
import json
import random
from typing import List, cast
from openai import NOT_GIVEN, OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .config import LLMCfg, ModelCfg

class LLMEngine:
    """LLM engine with conversation management and internal metric tracking."""
    
    def __init__(self, llm_cfg: LLMCfg, client: OpenAI) -> None:
        self.llm_cfg = llm_cfg
        self.system_prompt = llm_cfg.system_prompt
        self.client = client
        self.selected_model: ModelCfg = None
        
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

    def generate(self, prompt: str) -> str:
        # Select a new model for each generation (optional - could be per call)
        self._select_model()
        
        # Add the user prompt to the conversation
        self.add_message("user", prompt)
        
        # Track response time
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.selected_model.name,
            messages=self.messages,
            temperature=self.selected_model.temperature if self.selected_model.temperature else NOT_GIVEN,
            timeout=self.selected_model.llm_timeout
        )

        response_time = time.time() - start_time
        
        content = response.choices[0].message.content
        
        # Extract token usage information
        usage = response.usage
        total_tokens = usage.total_tokens if usage else 0
        
        # Update internal metrics
        self._total_llm_time += response_time
        self._total_tokens += total_tokens
        
        # Add the assistant's response to the conversation
        if content:
            self.add_message("assistant", content)
        
        return content.strip() if content else ""
    
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
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_cfg.provider}")