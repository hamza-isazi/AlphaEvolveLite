import os
import time
import json
from typing import List, cast, Tuple
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .config import LLMCfg

class LLMEngine:
    """LLM engine with conversation management."""
    
    def __init__(self, llm_cfg: LLMCfg, client: OpenAI) -> None:
        self.model = llm_cfg.model
        self.system_prompt = llm_cfg.system_prompt
        self.temperature = llm_cfg.temperature
        self.client = client
            
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