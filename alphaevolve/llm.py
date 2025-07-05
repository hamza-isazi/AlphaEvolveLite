import os
from typing import Protocol
from openai import OpenAI

class LLMEngine(Protocol):
    def generate(self, prompt: str) -> str: ...


class OpenAIEngine:
    def __init__(self, model: str):
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in your environment.")
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()