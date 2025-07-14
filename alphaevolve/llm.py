import os
from typing import Protocol, List, Dict, Any, cast
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .config import LLMCfg

class LLMEngine(Protocol):
    def generate(self, prompt: str) -> str: ...
    def add_message(self, role: str, content: str) -> None: ...
    def reset_conversation(self) -> None: ...


class OpenAIEngine:
    def __init__(self, llm_cfg: LLMCfg) -> None:
        self.model = llm_cfg.model
        self.system_prompt = llm_cfg.system_prompt
        self.temperature = llm_cfg.temperature
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in your environment.")
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

    def generate(self, prompt: str) -> str:
        # Add the user prompt to the conversation
        self.add_message("user", prompt)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature
        )
        content = response.choices[0].message.content
        
        # Add the assistant's response to the conversation
        if content:
            self.add_message("assistant", content)
        
        return content.strip() if content else ""