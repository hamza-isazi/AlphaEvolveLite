from threading import Lock
import os
from openai import OpenAI


class ClientPool:
    def __init__(self):
        self._lock = Lock()
        self._cache: dict[str, OpenAI] = {}

    def _build(self, provider: str) -> OpenAI:
        p = provider.lower()
        if p == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Set OPENAI_API_KEY in your environment.")
            return OpenAI(api_key=api_key)
        if p == "google":
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("Set GOOGLE_API_KEY in your environment.")
            return OpenAI(api_key=api_key,
                          base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        if p == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("Set OPENROUTER_API_KEY in your environment.")
            return OpenAI(api_key=api_key,
                          base_url="https://openrouter.ai/api/v1")
        raise ValueError(f"Unsupported LLM provider: {provider}")

    def get(self, provider: str) -> OpenAI:
        key = provider.lower()
        cli = self._cache.get(key)
        if cli is not None:
            return cli
        with self._lock:
            cli = self._cache.get(key)
            if cli is None:
                cli = self._build(provider)
                self._cache[key] = cli
            return cli


_GLOBAL_POOL = ClientPool()


def global_client_pool() -> ClientPool:
    return _GLOBAL_POOL