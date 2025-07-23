import os
import time
import json
from typing import Protocol, List, cast, Tuple
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .config import LLMCfg

class LLMEngine(Protocol):
    def generate(self, prompt: str) -> Tuple[str, float, int]: ...
    def add_message(self, role: str, content: str) -> None: ...
    def reset_conversation(self) -> None: ...
    def get_conversation_json(self) -> str: ...


class BaseLLMEngine:
    """Base class for LLM engines with shared functionality."""
    
    def __init__(self, llm_cfg: LLMCfg, api_key_env: str, base_url: str = None) -> None:
        self.model = llm_cfg.model
        self.system_prompt = llm_cfg.system_prompt
        self.temperature = llm_cfg.temperature
        
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {api_key_env} in your environment.")
        
        # Initialize client with optional base URL
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
            
        self.messages: List[ChatCompletionMessageParam] = [
            cast(ChatCompletionMessageParam, {"role": "system", "content": self.system_prompt})
        ]

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

    def generate(self, prompt: str) -> Tuple[str, float, int]:
        # Add the user prompt to the conversation
        self.add_message("user", prompt)
        
        # Track response time
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature
        )
        response_time = time.time() - start_time
        
        content = response.choices[0].message.content
        
        # Extract token usage information
        usage = response.usage
        total_tokens = usage.total_tokens if usage else 0
        
        # Add the assistant's response to the conversation
        if content:
            self.add_message("assistant", content)
        
        return (content.strip() if content else "", response_time, total_tokens)


class OpenAIEngine(BaseLLMEngine):
    def __init__(self, llm_cfg: LLMCfg) -> None:
        super().__init__(llm_cfg, "OPENAI_API_KEY")


class GeminiEngine(BaseLLMEngine):
    def __init__(self, llm_cfg: LLMCfg) -> None:
        super().__init__(
            llm_cfg, 
            "GOOGLE_API_KEY", 
            "https://generativelanguage.googleapis.com/v1beta/openai/"
        )


def create_llm_engine(llm_cfg: LLMCfg) -> LLMEngine:
    """Factory function to create the appropriate LLM engine based on provider."""
    if llm_cfg.provider.lower() == "openai":
        return OpenAIEngine(llm_cfg)
    elif llm_cfg.provider.lower() == "gemini":
        return GeminiEngine(llm_cfg)
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_cfg.provider}")