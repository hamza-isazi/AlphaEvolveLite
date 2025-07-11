import os
from typing import Protocol
from openai import OpenAI

from .config import LLMCfg

class LLMEngine(Protocol):
    def generate(self, prompt: str) -> str: ...


class OpenAIEngine:
    def __init__(self, llm_cfg: LLMCfg) -> None:
        self.model = llm_cfg.model
        self.system_prompt = llm_cfg.system_prompt
        self.temperature = llm_cfg.temperature
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in your environment.")
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(model=self.model,
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=self.temperature)
        content = response.choices[0].message.content
        return content.strip() if content else ""